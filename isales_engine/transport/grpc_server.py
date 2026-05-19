"""Production :class:`CloudEdgeServer` backed by grpc.aio.

Spec: service-communication § Requirement: 云-边控制面 (cloud-edge gRPC
bidirectional streaming);
      architecture § Requirement: Engine session 与 cloud instance co-location.

One :class:`CloudEdgeGrpcServer` instance per engine process. It fans out
to N edge connections, one bidi stream per edge identified by its
``edge_device_id`` (derived from the Bearer token via :class:`TokenVerifier`).
Engine session code calls :meth:`CloudEdgeGrpcServer.send_to_edge` to push a
``Cloud2Edge`` frame to a named edge; inbound ``Edge2Cloud`` frames arrive
via the callback registered with :meth:`on_edge_message`.

Reconnect semantics: if the same ``edge_device_id`` opens a new stream
while an old one is still tracked, the old stream is force-closed and
displaced. This keeps the "one logical edge → one active stream" invariant
without depending on the edge to cleanly tear down before reconnecting.

Buffering: inbound frames are NOT buffered server-side — the registered
callback fires synchronously per frame (any exception is logged and the
stream continues, since the cloud is the source-of-truth and dropping one
edge event isn't catastrophic). Outbound frames have a per-stream
``asyncio.Queue`` for ordered delivery.

Auth: Bearer token in gRPC initial metadata, verified once at stream open
via :class:`TokenVerifier`. Token failures close the stream with
``UNAUTHENTICATED``; successful verification binds the ``EdgeIdentity`` to
the stream for the duration of the connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
import grpc.aio
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.proto import cloud_edge_pb2_grpc
from isales_common.transport.cloud_edge import (
    CloudEdgeServer,
    EdgeIdentity,
    EdgeMessageCallback,
    EdgeNotConnected,
    InvalidToken,
    TokenVerifier,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

__all__ = ["CloudEdgeGrpcServer"]


#: HTTP/2 keepalive server options. Symmetric to the client-side options
#: in isales-telephony's grpc_client. Server SHALL accept the edge's 30 s
#: keepalive pings without striking them as "too frequent" (default gRPC
#: server policy boots client streams that ping more often than every
#: 5 minutes). Cloud-edge is a low-fanout, fully trusted-by-JWT channel,
#: so the anti-DDoS defaults are wrong here.
#: Spec: openspec/changes/cloud-edge-grpc-keepalive/.
_KEEPALIVE_SERVER_OPTIONS: list[tuple[str, int]] = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_ping_strikes", 0),
    ("grpc.http2.min_time_between_pings_ms", 10000),
    ("grpc.http2.min_ping_interval_without_data_ms", 10000),
]


# Sentinel pushed onto an outbound queue to signal "drain and close".
# Using a typed sentinel rather than None lets the outbound iterator
# distinguish between "no item yet" and "shut down".
_CLOSE = object()


class _ServerStream:
    """Per-connection state on the server side.

    Holds the identity, an outbound queue (engine session → edge), and a
    ``_closed`` flag. Pre-close, :meth:`send` accepts; post-close, it
    raises :class:`EdgeNotConnected`.

    The outbound iterator terminates cleanly when :meth:`close` is invoked
    by pushing the :data:`_CLOSE` sentinel.
    """

    def __init__(self, identity: EdgeIdentity) -> None:
        self.identity = identity
        # Bounded queue — 64 messages is generous for cloud → edge control
        # plane (dial / cancel / heartbeat). If we ever fill this, the
        # engine session is producing faster than the network can drain,
        # which means something is wrong upstream.
        self._outbound: asyncio.Queue[pb.Cloud2Edge | object] = asyncio.Queue(
            maxsize=64,
        )
        self._closed = False

    @property
    def connected(self) -> bool:
        return not self._closed

    async def send(self, message: pb.Cloud2Edge) -> None:
        if self._closed:
            raise EdgeNotConnected(self.identity.edge_device_id)
        await self._outbound.put(message)

    async def outbound(self) -> AsyncIterator[pb.Cloud2Edge]:
        while True:
            item = await self._outbound.get()
            if item is _CLOSE:
                return
            assert isinstance(item, pb.Cloud2Edge)
            yield item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Best-effort sentinel push. If the queue is full, the outbound
        # consumer will hit the closed flag on its next iteration via the
        # connected check; this prevents close from blocking.
        try:
            self._outbound.put_nowait(_CLOSE)
        except asyncio.QueueFull:
            logger.warning(
                "outbound queue full at close; relying on connection drop",
            )


class _Servicer(cloud_edge_pb2_grpc.CloudEdgeServicer):
    """Thin grpc.aio servicer that delegates to :class:`CloudEdgeGrpcServer`.

    Kept separate from the public server class so callers can't accidentally
    bypass the abstract :class:`CloudEdgeServer` surface.
    """

    def __init__(self, parent: CloudEdgeGrpcServer) -> None:
        self._parent = parent

    async def Bidi(  # noqa: N802 — matches generated stub name
        self,
        request_iterator: AsyncIterator[pb.Edge2Cloud],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.Cloud2Edge]:
        async for outbound in self._parent._handle_bidi(  # noqa: SLF001
            request_iterator,
            context,
        ):
            yield outbound


class CloudEdgeGrpcServer(CloudEdgeServer):
    """Production :class:`CloudEdgeServer` implementation.

    Construction is decoupled from listening; create the server, register
    callbacks, then call :meth:`start` to begin accepting RPCs. Tests can
    reuse the in-memory implementation from :mod:`isales_common.transport.
    testing`; this class is for integration tests + production.
    """

    def __init__(self, *, token_verifier: TokenVerifier) -> None:
        self._verifier = token_verifier
        self._edge_callback: EdgeMessageCallback | None = None
        self._streams: dict[str, _ServerStream] = {}
        # Guards _streams + the reconnect race (concurrent connects from
        # the same edge_device_id displacing each other).
        self._streams_lock = asyncio.Lock()
        self._server: grpc.aio.Server | None = None

    # ----- CloudEdgeServer surface ---------------------------------------

    async def start(
        self,
        listen_addr: str,
        *,
        server_credentials: grpc.ServerCredentials | None = None,
        interceptors: Iterable[grpc.aio.ServerInterceptor] | None = None,
    ) -> None:
        if self._server is not None:
            raise RuntimeError("server already started")
        servicer = _Servicer(self)
        self._server = grpc.aio.server(
            interceptors=list(interceptors or []),
            options=_KEEPALIVE_SERVER_OPTIONS,
        )
        cloud_edge_pb2_grpc.add_CloudEdgeServicer_to_server(  # type: ignore[no-untyped-call]
            servicer,
            self._server,
        )
        if server_credentials is None:
            self._server.add_insecure_port(listen_addr)
        else:
            self._server.add_secure_port(listen_addr, server_credentials)
        await self._server.start()
        logger.info("cloud_edge_grpc_server_started", extra={"addr": listen_addr})

    async def stop(self, grace_seconds: float = 5.0) -> None:
        # Drain all attached streams first so their outbound coroutines
        # exit cleanly; grpcio's server.stop(grace=...) waits for in-flight
        # RPCs to wind down.
        async with self._streams_lock:
            for stream in list(self._streams.values()):
                stream.close()
        if self._server is not None:
            await self._server.stop(grace=grace_seconds)
            self._server = None

    async def send_to_edge(
        self,
        edge_device_id: str,
        message: pb.Cloud2Edge,
    ) -> None:
        async with self._streams_lock:
            stream = self._streams.get(edge_device_id)
        if stream is None or not stream.connected:
            raise EdgeNotConnected(edge_device_id)
        await stream.send(message)

    def on_edge_message(self, callback: EdgeMessageCallback) -> None:
        self._edge_callback = callback

    # ----- internal: bidi RPC handler -----------------------------------

    async def _handle_bidi(
        self,
        request_iterator: AsyncIterator[pb.Edge2Cloud],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.Cloud2Edge]:
        """The actual servicer body — authenticate, register the stream,
        spawn a reader, then yield outbound frames until the stream closes.
        """
        identity = await self._authenticate(context)
        if identity is None:
            # Authenticate already aborted the context.
            return

        # Bidi servers don't normally yield until they have something to
        # send, but the grpc.aio client uses `call.initial_metadata()` to
        # know the RPC was accepted. Flush metadata eagerly so the edge's
        # CloudEdgeGrpcClient.start() can race-free transition to
        # is_connected=True instead of waiting for an arbitrary first
        # response.
        await context.send_initial_metadata([])
        logger.info(
            "cloud_edge_stream_opened",
            extra={"edge_device_id": identity.edge_device_id},
        )

        stream = _ServerStream(identity)
        await self._register_stream(stream)

        reader_task = asyncio.create_task(
            self._reader_loop(identity, stream, request_iterator),
            name=f"cloud_edge_reader_{identity.edge_device_id}",
        )
        try:
            async for outbound in stream.outbound():
                yield outbound
        finally:
            # Whether we exited cleanly (close sentinel) or by client
            # disconnect (request_iterator raised in reader_task), make
            # sure we tear down both sides + unregister.
            stream.close()
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader_task
            await self._deregister_stream(identity.edge_device_id, stream)

    async def _authenticate(
        self,
        context: grpc.aio.ServicerContext,
    ) -> EdgeIdentity | None:
        metadata = dict(context.invocation_metadata() or [])
        auth = metadata.get("authorization", "")
        if not auth.startswith("Bearer "):
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "missing or malformed Bearer token in `authorization` metadata",
            )
            return None
        token = auth.removeprefix("Bearer ").strip()
        try:
            return await self._verifier.verify(token)
        except InvalidToken as exc:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
            return None

    async def _register_stream(self, stream: _ServerStream) -> None:
        edge_id = stream.identity.edge_device_id
        async with self._streams_lock:
            old = self._streams.get(edge_id)
            if old is not None:
                # Same edge reconnecting (probably auto-reconnect after the
                # client lost the stream). Close the stale one so any
                # outstanding send_to_edge calls fail loudly rather than
                # silently routing to a dead connection.
                old.close()
                logger.info(
                    "cloud_edge_stream_displaced",
                    extra={"edge_device_id": edge_id},
                )
            self._streams[edge_id] = stream

    async def _deregister_stream(self, edge_id: str, stream: _ServerStream) -> None:
        async with self._streams_lock:
            if self._streams.get(edge_id) is stream:
                del self._streams[edge_id]

    async def _reader_loop(
        self,
        identity: EdgeIdentity,
        stream: _ServerStream,
        request_iterator: AsyncIterator[pb.Edge2Cloud],
    ) -> None:
        """Dispatch inbound Edge2Cloud frames to the registered callback.

        The reader terminates when:

        - the client closes the stream (``request_iterator`` exits cleanly), or
        - the underlying transport drops (``request_iterator`` raises), or
        - this task is cancelled (the outer ``_handle_bidi`` exited).

        In all cases we close the per-stream outbound queue so the outer
        Bidi async generator finishes its ``async for`` and the gRPC
        framework can complete the RPC.
        """
        try:
            async for msg in request_iterator:
                if self._edge_callback is None:
                    continue
                try:
                    await self._edge_callback(identity, msg)
                except Exception:  # noqa: BLE001 — handler bugs shouldn't kill the stream
                    logger.exception(
                        "edge_callback raised; continuing stream",
                        extra={"edge_device_id": identity.edge_device_id},
                    )
        finally:
            stream.close()
