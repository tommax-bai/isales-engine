"""Stage-6 RealTelephonyClient — Unix socket bridge to isales-telephony's modem-controller.

Spec: device-hardware § engine ↔ modem-controller IPC 协议. NDJSON frames
over a Unix socket; engine sends ``cmd``, modem-controller emits
``event``. Sessions correlate via the ``session_id`` field which we mint
from the engine-side ``int`` call id (stringified).

Non-goals here:
- The audio loop (``audio_in``/``audio_out``) is implemented but kept
  minimal — actual GSM PCM frames flow once a real device backend is
  wired in PR #11. The contract & event routing are what matters now.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from isales_engine.realtime.telephony_client import TelephonyClient, TelephonyEvent

logger = logging.getLogger(__name__)

_SENTINEL: Any = object()


class TelephonyDialFailed(Exception):
    """Modem-controller refused the dial or hung up before connecting."""


class TelephonyDeviceError(Exception):
    """Underlying device emitted device_error or the IPC link dropped."""


class RealTelephonyClient(TelephonyClient):
    """Bridge engine state machine to modem-controller IPC.

    One client owns one Unix socket connection and multiplexes per-session
    event/audio queues. The protocol is defined by device-hardware spec §
    engine ↔ modem-controller IPC 协议 + the v1 telephony handlers (which
    additionally include ``call_id`` + ``device_id`` fields for backward
    compat with stage-2 clients).
    """

    def __init__(
        self,
        socket_path: str,
        *,
        dial_timeout_s: float = 60.0,
    ) -> None:
        self._socket_path = socket_path
        self._dial_timeout_s = dial_timeout_s
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connect_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        # Per-session queues, keyed by stringified engine call_id.
        self._events: dict[str, asyncio.Queue[TelephonyEvent | object]] = {}
        self._inbound: dict[str, asyncio.Queue[bytes | object]] = {}
        # Per-session metadata supplied at dial time; surfaced to the
        # protocol when sending hangup / audio_downstream so the modem can
        # route by session_id alone.
        self._device_ids: dict[str, int] = {}

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._writer is not None:
                return
            self._reader, self._writer = await asyncio.open_unix_connection(
                self._socket_path
            )
            self._reader_task = asyncio.create_task(
                self._read_loop(), name="real_telephony_reader"
            )
            logger.info("real_telephony_connected", extra={"path": self._socket_path})

    async def aclose(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

    # ---- TelephonyClient interface ------------------------------------

    async def dial(self, call_id: int, phone: str) -> None:
        """Dial via the modem-controller; emits 'connected' or hangup on the events stream."""

        await self.connect()
        sid = str(call_id)
        self._ensure_session(sid)
        # Engine's contract doesn't currently surface device_id at the
        # client level; the scheduler-side /devices/select hands one to
        # engine in the dial message. For v1 we accept a missing device id
        # — the modem-controller picks the first idle device.
        device_id = self._device_ids.get(sid, 0)
        await self._write({
            "cmd": "dial",
            "session_id": sid,
            "device_id": device_id,
            "number": phone,
        })

    async def hangup(self, call_id: int) -> None:
        await self.connect()
        sid = str(call_id)
        await self._write({"cmd": "hangup", "session_id": sid})
        # Surface the local hangup synthetically; the modem will also
        # emit a remote_hangup which we discard if local already fired.
        await self._events.setdefault(sid, asyncio.Queue()).put(
            TelephonyEvent(type="local_hangup", call_id=call_id)
        )
        self._close_session(sid)

    def audio_in(self, call_id: int) -> AsyncIterator[bytes]:
        sid = str(call_id)
        self._ensure_session(sid)

        async def _iter() -> AsyncIterator[bytes]:
            q = self._inbound[sid]
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    return
                assert isinstance(item, bytes)
                yield item

        return _iter()

    async def audio_out(self, call_id: int, chunks: AsyncIterator[bytes]) -> None:
        sid = str(call_id)
        async for chunk in chunks:
            await self._write({
                "cmd": "audio_downstream",
                "session_id": sid,
                "pcm_chunk": base64.b64encode(chunk).decode("ascii"),
            })

    def events(self, call_id: int) -> AsyncIterator[TelephonyEvent]:
        sid = str(call_id)
        self._ensure_session(sid)

        async def _iter() -> AsyncIterator[TelephonyEvent]:
            q = self._events[sid]
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    return
                assert isinstance(item, TelephonyEvent)
                yield item

        return _iter()

    # ---- internals ----------------------------------------------------

    def set_device_for_session(self, call_id: int, device_id: int) -> None:
        """Optional hint: the dial command will carry device_id when set."""

        self._device_ids[str(call_id)] = device_id

    def _ensure_session(self, sid: str) -> None:
        self._events.setdefault(sid, asyncio.Queue())
        self._inbound.setdefault(sid, asyncio.Queue())

    def _close_session(self, sid: str) -> None:
        if sid in self._inbound:
            self._inbound[sid].put_nowait(_SENTINEL)
        if sid in self._events:
            self._events[sid].put_nowait(_SENTINEL)

    async def _write(self, msg: dict[str, Any]) -> None:
        assert self._writer is not None, "RealTelephonyClient.connect() not called"
        line = json.dumps(msg, ensure_ascii=False).encode() + b"\n"
        self._writer.write(line)
        await self._writer.drain()

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                raw = await self._reader.readline()
                if not raw:
                    self._broadcast_disconnect()
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "real_telephony_bad_frame", extra={"raw": repr(raw[:120])}
                    )
                    continue
                if not isinstance(msg, dict):
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("real_telephony_reader_failed")
            self._broadcast_disconnect()

    def _dispatch(self, msg: dict[str, Any]) -> None:
        event = msg.get("event")
        if not isinstance(event, str):
            return
        sid = msg.get("session_id")
        if not isinstance(sid, str):
            # Per spec § engine ↔ modem-controller IPC 协议, every event
            # MUST carry session_id. A frame without one is malformed;
            # engine can't route it anyway, so drop and warn.
            logger.warning("real_telephony_event_missing_session_id", extra={"msg": msg})
            return
        self._ensure_session(sid)
        try:
            engine_call_id = int(sid)
        except ValueError:
            return

        if event == "dial_ack":
            return  # Synchronous ack; no engine-side state change.

        if event == "connected":
            self._events[sid].put_nowait(
                TelephonyEvent(type="connected", call_id=engine_call_id)
            )
            return

        if event == "remote_hangup":
            self._events[sid].put_nowait(
                TelephonyEvent(
                    type="remote_hangup",
                    call_id=engine_call_id,
                    detail=str(msg.get("cause") or "remote_clearing"),
                )
            )
            self._close_session(sid)
            return

        if event == "device_error":
            self._events[sid].put_nowait(
                TelephonyEvent(
                    type="device_error",
                    call_id=engine_call_id,
                    detail=str(msg.get("code") or msg.get("message") or ""),
                )
            )
            self._close_session(sid)
            return

        if event == "audio_upstream":
            chunk_b64 = msg.get("pcm_chunk")
            if isinstance(chunk_b64, str):
                try:
                    chunk = base64.b64decode(chunk_b64)
                except Exception:
                    return
                self._inbound[sid].put_nowait(chunk)
            return

        # call_progress, hangup_ack, etc. are informational; no engine state change.

    def _broadcast_disconnect(self) -> None:
        for sid, queue in self._events.items():
            try:
                engine_call_id = int(sid)
            except ValueError:
                continue
            queue.put_nowait(
                TelephonyEvent(
                    type="device_error",
                    call_id=engine_call_id,
                    detail="ipc_disconnected",
                )
            )
            queue.put_nowait(_SENTINEL)
        for q in self._inbound.values():
            q.put_nowait(_SENTINEL)
