"""Integration tests for :class:`CloudEdgeGrpcServer`.

Spec: service-communication § Requirement: 云-边控制面 (cloud-edge gRPC
bidirectional streaming).

These spin up a real grpc.aio server on a free localhost port, connect a
real grpc.aio insecure channel, and exchange real protobuf frames. The
production server-side implementation is exercised end-to-end against the
generated stub — only the edge-side ``CloudEdgeClient`` impl (which lives
in isales-telephony) is omitted; we use a thin asyncio client helper
instead so this test suite has no out-of-repo runtime deps.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import grpc
import grpc.aio
import pytest
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.proto import cloud_edge_pb2_grpc
from isales_common.transport.cloud_edge import (
    EdgeIdentity,
    EdgeNotConnected,
)
from isales_common.transport.testing import StaticTokenVerifier

from isales_engine.transport.grpc_server import CloudEdgeGrpcServer

# --------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------


class _GrpcEdgeProbe:
    """Minimal asyncio gRPC client for driving the server in tests.

    Wraps a single bidi stream — sends Edge2Cloud frames via :meth:`send`
    and surfaces Cloud2Edge frames via :meth:`recv` (or iterates via
    :meth:`responses`).

    NOT a substitute for the production CloudEdgeClient impl that lives in
    isales-telephony; this is just a test driver scoped to the server
    under test.
    """

    def __init__(self, target: str, token: str | None) -> None:
        self._target = target
        self._token = token
        self._channel: grpc.aio.Channel | None = None
        self._call: grpc.aio.StreamStreamCall | None = None
        # asyncio.Queue of outbound frames the test will send.
        self._outbound: asyncio.Queue[pb.Edge2Cloud | None] = asyncio.Queue()
        # asyncio.Queue of inbound frames received from the server.
        self._inbound: asyncio.Queue[pb.Cloud2Edge] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _GrpcEdgeProbe:
        self._channel = grpc.aio.insecure_channel(self._target)
        stub = cloud_edge_pb2_grpc.CloudEdgeStub(self._channel)
        metadata: list[tuple[str, str]] = []
        if self._token is not None:
            metadata.append(("authorization", f"Bearer {self._token}"))
        self._call = stub.Bidi(self._outbound_iter(), metadata=metadata)
        self._reader_task = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def _outbound_iter(self) -> AsyncIterator[pb.Edge2Cloud]:
        while True:
            msg = await self._outbound.get()
            if msg is None:
                return
            yield msg

    async def _read_loop(self) -> None:
        assert self._call is not None
        try:
            async for response in self._call:
                await self._inbound.put(response)
        except grpc.aio.AioRpcError:
            # Surface via inbound queue closure; tests assert via the call object.
            pass

    async def send(self, msg: pb.Edge2Cloud) -> None:
        await self._outbound.put(msg)

    async def recv(self, timeout: float = 2.0) -> pb.Cloud2Edge:  # noqa: ASYNC109
        return await asyncio.wait_for(self._inbound.get(), timeout=timeout)

    async def expect_rpc_error(self, timeout: float = 2.0) -> grpc.aio.AioRpcError:  # noqa: ASYNC109, ARG002
        """Wait for the server to abort the call; return the error."""
        assert self._call is not None
        try:
            async for _ in self._call:
                pass
        except grpc.aio.AioRpcError as e:
            return e
        else:
            raise AssertionError("expected the call to be aborted by the server")

    async def close(self) -> None:
        await self._outbound.put(None)
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
        if self._channel is not None:
            await self._channel.close()


async def _start_server(
    *,
    token: str = "good-token",
    edge_id: str = "edge-1",
) -> tuple[CloudEdgeGrpcServer, str]:
    verifier = StaticTokenVerifier(token=token, edge_device_id=edge_id)
    server = CloudEdgeGrpcServer(token_verifier=verifier)
    # Listen on an arbitrary free port; grpcio returns the actual port.
    # We use add_insecure_port directly via start() with a 0-port string.
    # grpcio sets the listening port internally; for tests we resolve it
    # by inspecting the inner server. Simpler approach: pick a known free
    # port via a temporary socket bind.
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    await server.start(f"127.0.0.1:{port}")
    return server, f"127.0.0.1:{port}"


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_then_stop_clean() -> None:
    server, _ = await _start_server()
    await server.stop(grace_seconds=0.1)


@pytest.mark.asyncio
async def test_double_start_raises() -> None:
    server, _ = await _start_server()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await server.start("127.0.0.1:1")
    finally:
        await server.stop(grace_seconds=0.1)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_bearer_token_unauthenticated() -> None:
    server, target = await _start_server()
    try:
        async with _GrpcEdgeProbe(target, token=None) as probe:
            err = await probe.expect_rpc_error()
            assert err.code() == grpc.StatusCode.UNAUTHENTICATED
            assert "Bearer" in (err.details() or "")
    finally:
        await server.stop(grace_seconds=0.1)


@pytest.mark.asyncio
async def test_invalid_token_unauthenticated() -> None:
    server, target = await _start_server(token="right", edge_id="edge-1")
    try:
        async with _GrpcEdgeProbe(target, token="wrong") as probe:
            err = await probe.expect_rpc_error()
            assert err.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        await server.stop(grace_seconds=0.1)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_call_event_invokes_callback() -> None:
    server, target = await _start_server()
    received: list[tuple[EdgeIdentity, pb.Edge2Cloud]] = []
    event = asyncio.Event()

    async def handler(identity: EdgeIdentity, msg: pb.Edge2Cloud) -> None:
        received.append((identity, msg))
        event.set()

    server.on_edge_message(handler)

    try:
        async with _GrpcEdgeProbe(target, token="good-token") as probe:
            await probe.send(
                pb.Edge2Cloud(
                    call_event=pb.CallEvent(call_id="c-1", connected=pb.Connected()),
                ),
            )
            await asyncio.wait_for(event.wait(), timeout=2.0)

        assert len(received) == 1
        identity, msg = received[0]
        assert identity == EdgeIdentity(edge_device_id="edge-1")
        assert msg.call_event.WhichOneof("kind") == "connected"
    finally:
        await server.stop(grace_seconds=0.1)


@pytest.mark.asyncio
async def test_send_to_edge_delivers_cloud2edge_to_client() -> None:
    server, target = await _start_server()
    try:
        async with _GrpcEdgeProbe(target, token="good-token") as probe:
            # Drive at least one inbound frame so we know the server has
            # registered the stream before we call send_to_edge.
            await probe.send(pb.Edge2Cloud(heartbeat=pb.Heartbeat()))
            await asyncio.sleep(0.05)  # let server _register_stream run

            await server.send_to_edge(
                "edge-1",
                pb.Cloud2Edge(cancel=pb.CancelCommand(call_id="c-1", reason="test")),
            )
            response = await probe.recv()
            assert response.WhichOneof("payload") == "cancel"
            assert response.cancel.call_id == "c-1"
    finally:
        await server.stop(grace_seconds=0.1)


@pytest.mark.asyncio
async def test_send_to_unconnected_edge_raises() -> None:
    server, _ = await _start_server()
    try:
        with pytest.raises(EdgeNotConnected):
            await server.send_to_edge(
                "no-such-edge",
                pb.Cloud2Edge(heartbeat=pb.Heartbeat()),
            )
    finally:
        await server.stop(grace_seconds=0.1)


# --------------------------------------------------------------------------
# Reconnect / stream displacement
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_displaces_old_stream() -> None:
    """Same edge_device_id reconnecting closes the prior stream.

    Cloud-side state is the source of truth; the old TCP connection may
    be wedged, and we don't want send_to_edge silently routing to it.
    """
    server, target = await _start_server()
    try:
        async with _GrpcEdgeProbe(target, token="good-token") as first:
            await first.send(pb.Edge2Cloud(heartbeat=pb.Heartbeat()))
            await asyncio.sleep(0.05)

            # Second probe with same token (= same edge_device_id).
            async with _GrpcEdgeProbe(target, token="good-token") as second:
                await second.send(pb.Edge2Cloud(heartbeat=pb.Heartbeat()))
                await asyncio.sleep(0.05)

                # send_to_edge should land on the *new* stream.
                await server.send_to_edge(
                    "edge-1",
                    pb.Cloud2Edge(cancel=pb.CancelCommand(call_id="c-2", reason="new")),
                )
                response = await second.recv()
                assert response.cancel.call_id == "c-2"
    finally:
        await server.stop(grace_seconds=0.1)


# --------------------------------------------------------------------------
# Callback exception isolation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_exception_does_not_kill_stream() -> None:
    """Buggy callback raising MUST NOT close the stream — the engine
    needs to keep receiving edge events even if one handler invocation
    blew up."""
    server, target = await _start_server()
    call_count = 0
    second_received = asyncio.Event()

    async def handler(_identity: EdgeIdentity, msg: pb.Edge2Cloud) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("test-induced handler bug")
        second_received.set()

    server.on_edge_message(handler)

    try:
        async with _GrpcEdgeProbe(target, token="good-token") as probe:
            await probe.send(
                pb.Edge2Cloud(
                    call_event=pb.CallEvent(call_id="c-1", ringing=pb.Ringing()),
                ),
            )
            await asyncio.sleep(0.05)
            await probe.send(
                pb.Edge2Cloud(
                    call_event=pb.CallEvent(call_id="c-1", connected=pb.Connected()),
                ),
            )
            await asyncio.wait_for(second_received.wait(), timeout=2.0)

        assert call_count == 2
    finally:
        await server.stop(grace_seconds=0.1)


# ---------------------------------------------------------------------------
# cloud-edge-grpc-keepalive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_constructed_with_keepalive_options(
    monkeypatch,
) -> None:
    """Spec: cloud-edge-grpc-keepalive § "cloud gRPC server 对称配置不踢
    client ping". Refactor-guard: ensures the 6 keepalive entries stay
    on the server's grpc.aio.server() options.
    """
    captured: list[dict[str, object]] = []
    real_server = grpc.aio.server

    def _spy_server(*args, **kwargs):
        captured.append({"args": args, **kwargs})
        return real_server(*args, **kwargs)

    monkeypatch.setattr(grpc.aio, "server", _spy_server)

    server, _ = await _start_server()
    try:
        assert captured, "grpc.aio.server() was not invoked"
        options = dict(captured[0].get("options") or [])
        assert options.get("grpc.keepalive_time_ms") == 30000
        assert options.get("grpc.keepalive_timeout_ms") == 10000
        assert options.get("grpc.keepalive_permit_without_calls") == 1
        assert options.get("grpc.http2.max_ping_strikes") == 0
        assert options.get("grpc.http2.min_time_between_pings_ms") == 10000
        assert options.get(
            "grpc.http2.min_ping_interval_without_data_ms",
        ) == 10000
    finally:
        await server.stop(grace_seconds=0.1)


@pytest.mark.asyncio
async def test_stream_opened_log_fires_after_send_initial_metadata(
    caplog,
) -> None:
    """Spec: cloud-edge-grpc-keepalive § "stream 上线 + 上线时分别打 INFO 日志".

    Server SHALL log ``cloud_edge_stream_opened`` with the resolved
    ``edge_device_id`` extra, right after ``send_initial_metadata`` and
    before the stream is registered. Lets dev/ops correlate against the
    client's matching ``cloud_edge_stream_connected``.
    """
    import logging

    caplog.set_level(
        logging.INFO, logger="isales_engine.transport.grpc_server",
    )
    server, target = await _start_server(edge_id="edge-keepalive-test")
    try:
        async with _GrpcEdgeProbe(target, token="good-token") as probe:
            await probe.send(
                pb.Edge2Cloud(heartbeat=pb.Heartbeat()),
            )
            await asyncio.sleep(0.1)
    finally:
        await server.stop(grace_seconds=0.1)

    opened_logs = [
        r for r in caplog.records
        if r.getMessage() == "cloud_edge_stream_opened"
    ]
    assert opened_logs, "expected cloud_edge_stream_opened INFO line"
    assert getattr(opened_logs[-1], "edge_device_id", None) == "edge-keepalive-test"
