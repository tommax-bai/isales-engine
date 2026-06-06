"""RealTelephonyClient end-to-end tests against an in-process fake socket.

The fake server replays the protocol shape isales-telephony's modem-controller
emits (cmd → event over NDJSON). It validates that:
- dial → connected event reaches the engine event stream
- audio_upstream frames reach audio_in()
- remote_hangup closes the session
- device_error surfaces and tears the session down
- IPC disconnect surfaces a synthetic device_error
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from isales_engine.realtime.real_telephony import RealTelephonyClient

# RealTelephonyClient is the single-host (Linux/macOS) transport: engine ↔
# modem-controller over a Unix domain socket. The fake server below relies on
# asyncio.start_unix_server, which does not exist on Windows. These tests run
# on the Linux/macOS single-host path; skip on win32 dev machines.
# Removal trigger: drop when the single-host Unix-socket transport is retired
# in favour of the cloud-edge gRPC/DingRTC path (real_telephony.py deleted).
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix domain sockets (asyncio.start_unix_server) unavailable on Windows",
)


class _FakeServer:
    """Minimal fake modem-controller — accepts cmd, emits scripted events."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.received: list[dict[str, object]] = []
        self._server: asyncio.AbstractServer | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self._on_dial: callable | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path)
        )

    async def stop(self) -> None:
        for w in self._writers:
            try:
                w.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = json.loads(line)
                self.received.append(msg)
                if msg.get("cmd") == "dial":
                    sid = msg["session_id"]
                    await self._send(
                        writer, {"event": "dial_ack", "session_id": sid}
                    )
                    if self._on_dial is not None:
                        await self._on_dial(writer, sid)
                elif msg.get("cmd") == "hangup":
                    sid = msg.get("session_id")
                    await self._send(
                        writer, {"event": "hangup_ack", "session_id": sid}
                    )
        except Exception:
            pass

    async def _send(self, writer: asyncio.StreamWriter, msg: dict[str, object]) -> None:
        writer.write(json.dumps(msg).encode() + b"\n")
        await writer.drain()

    def script_on_dial(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._on_dial = callback

    async def push_to_session(self, sid: str, msg: dict[str, object]) -> None:
        for w in self._writers:
            await self._send(w, {**msg, "session_id": sid})


def _socket_path() -> Path:
    fd, path = tempfile.mkstemp(suffix=".sock")
    os.close(fd)
    Path(path).unlink(missing_ok=True)
    return Path(path)


@pytest.fixture
async def server_and_client():  # type: ignore[no-untyped-def]
    path = _socket_path()
    server = _FakeServer(path)
    await server.start()
    client = RealTelephonyClient(str(path), dial_timeout_s=2.0)
    try:
        yield server, client
    finally:
        await client.aclose()
        await server.stop()
        path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dial_emits_connected_event(server_and_client) -> None:  # type: ignore[no-untyped-def]
    server, client = server_and_client

    async def script(writer, sid):  # type: ignore[no-untyped-def]
        await server._send(
            writer, {"event": "connected", "session_id": sid}
        )

    server.script_on_dial(script)

    await client.dial(42, "13800138000")
    events = client.events(42)
    evt = await asyncio.wait_for(events.__anext__(), timeout=2.0)
    assert evt.type == "connected"
    assert evt.call_id == 42

    sent = next(m for m in server.received if m.get("cmd") == "dial")
    assert sent["session_id"] == "42"
    assert sent["number"] == "13800138000"


@pytest.mark.asyncio
async def test_audio_upstream_chunks_reach_audio_in(server_and_client) -> None:  # type: ignore[no-untyped-def]
    server, client = server_and_client

    async def script(writer, sid):  # type: ignore[no-untyped-def]
        await server._send(
            writer, {"event": "connected", "session_id": sid}
        )
        for byte_block in (b"\x00\x01", b"\x02\x03"):
            await server._send(
                writer,
                {
                    "event": "audio_upstream",
                    "session_id": sid,
                    "pcm_chunk": base64.b64encode(byte_block).decode(),
                },
            )
        await server._send(writer, {"event": "remote_hangup", "session_id": sid})

    server.script_on_dial(script)
    await client.dial(7, "111")

    received: list[bytes] = []
    audio = client.audio_in(7)
    async for chunk in audio:
        received.append(chunk)
    assert received == [b"\x00\x01", b"\x02\x03"]


@pytest.mark.asyncio
async def test_audio_out_sends_base64_chunks(server_and_client) -> None:  # type: ignore[no-untyped-def]
    server, client = server_and_client

    async def script(writer, sid):  # type: ignore[no-untyped-def]
        await server._send(writer, {"event": "connected", "session_id": sid})

    server.script_on_dial(script)
    await client.dial(11, "111")

    async def chunks() -> AsyncIterator[bytes]:
        yield b"\xaa\xbb"
        yield b"\xcc\xdd"

    await client.audio_out(11, chunks())
    # Yield so the server-side reader coroutine drains the two frames.
    await asyncio.sleep(0.05)
    audio_msgs = [m for m in server.received if m.get("cmd") == "audio_downstream"]
    assert len(audio_msgs) == 2
    assert audio_msgs[0]["session_id"] == "11"
    assert base64.b64decode(audio_msgs[0]["pcm_chunk"]) == b"\xaa\xbb"


@pytest.mark.asyncio
async def test_device_error_event_closes_session(server_and_client) -> None:  # type: ignore[no-untyped-def]
    server, client = server_and_client

    async def script(writer, sid):  # type: ignore[no-untyped-def]
        await server._send(
            writer,
            {"event": "device_error", "session_id": sid, "code": "signal_lost"},
        )

    server.script_on_dial(script)
    await client.dial(99, "111")

    events = client.events(99)
    evt = await asyncio.wait_for(events.__anext__(), timeout=2.0)
    assert evt.type == "device_error"
    assert evt.detail == "signal_lost"
    # Stream closes after device_error.
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(events.__anext__(), timeout=2.0)


@pytest.mark.asyncio
async def test_socket_disconnect_surfaces_synthetic_device_error() -> None:
    path = _socket_path()
    server = _FakeServer(path)
    await server.start()
    client = RealTelephonyClient(str(path))

    async def script(writer, sid):  # type: ignore[no-untyped-def]
        # Drop the connection right after dial_ack — engine should see
        # device_error("ipc_disconnected").
        writer.close()

    server.script_on_dial(script)
    await client.dial(5, "111")
    events = client.events(5)
    evt = await asyncio.wait_for(events.__anext__(), timeout=2.0)
    assert evt.type == "device_error"
    assert evt.detail == "ipc_disconnected"

    await client.aclose()
    await server.stop()
    path.unlink(missing_ok=True)
