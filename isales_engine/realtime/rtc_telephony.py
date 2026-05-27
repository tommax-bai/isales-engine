"""RtcTelephonyClient — :class:`TelephonyClient` over cloud-edge gRPC + RTC.

Spec: arch-cloud-edge-split / design.md Decision 5 (control plane) +
Decision 2 (audio topology); device-hardware spec delta § Requirement:
云端 engine 的 ARTC SDK 接入.

The cloud-side replacement for v1's :class:`RealTelephonyClient` (which
talks to a same-host modem-controller over a Unix socket). In the cloud-
edge form factor, ``dial`` / ``hangup`` / events flow over the cloud-edge
gRPC control plane, and PCM flows over Aliyun RTC. This wrapper glues
together four pieces that already exist in isolation:

- :class:`CloudEdgeServer` (gRPC bidi to the edge) — for dial / cancel +
  inbound CallEvent / DialAck.
- :class:`EngineSessionDispatcher` (per-call routing of gRPC events).
- :class:`RtcSession` (one per call; pulls inbound PCM, pushes outbound).
- :class:`RtcTokenIssuer` (signs the engine + edge RTC tokens).

The existing engine state machine / pipeline / wrap-up code is written
against :class:`TelephonyClient`'s five-method contract; this wrapper
satisfies that contract so business code is unaware of the cloud-edge
split. v1 :class:`RealTelephonyClient` is preserved alongside (dual-mode
during the A2 rollout); a future change may retire it.

Per-edge binding: one ``RtcTelephonyClient`` instance corresponds to one
``edge_device_id``. For multi-edge deployments, the engine session
factory selects the right client per dial based on which edge owns the
device. (Scheduler-side device → edge mapping is out of scope for this
class.)

Per-call lifecycle (engine call_id, modeled as int per the existing
TelephonyClient interface; serialised as ``str(call_id)`` on the wire):

1. ``dial(call_id, phone)``:
   - Sign engine + edge RTC credentials.
   - Create a new RtcSession; join the engine side as ``"engine-{sid}"``.
   - Start a pump task draining inbound PCM into the per-call queue.
   - Register the call with the dispatcher.
   - Send Cloud2Edge.DialCommand carrying the edge token + uids + phone +
     optional device_id (hinted via :meth:`set_device_for_session`).
   - Return immediately. ``"connected"`` arrives via the events queue
     once the edge fires CallEvent.connected (same contract as
     RealTelephonyClient).

2. ``audio_in(call_id)``: per-call inbound bytes generator, fed by the
   pump task (filtered to PCM from the bound edge uid).
3. ``audio_out(call_id, chunks)``: forwards each chunk to
   :meth:`RtcSession.push_audio` with a monotonically increasing
   timestamp.
4. ``events(call_id)``: per-call TelephonyEvent generator. Sources:
   - ``connected`` from CallEvent.connected.
   - ``remote_hangup`` from CallEvent.remote_hangup (detail = canonical
     hangup_cause snake_case).
   - ``device_error`` from CallEvent.device_error.
   - ``local_hangup`` synthesised by :meth:`hangup`.
   - ``device_error`` synthesised when the gRPC stream is gone at dial
     time (``EdgeNotConnected``).

5. ``hangup(call_id)``: best-effort Cloud2Edge.CancelCommand + sentinel
   the iterators + leave the RTC channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from isales_common.audio.rtc import RtcNotJoined, RtcSession
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import (
    CloudEdgeServer,
    EdgeNotConnected,
)

from isales_engine.realtime.telephony_client import (
    TelephonyClient,
    TelephonyEvent,
)
from isales_engine.transport.rtc_token import RtcTokenIssuer
from isales_engine.transport.session_dispatcher import EngineSessionDispatcher

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-private sentinel for terminating audio_in / events iterators.
# Same pattern as :class:`RealTelephonyClient`; kept private to avoid an
# implicit cross-module dependency between two TelephonyClient impls.
_SENTINEL: Any = object()


_HANGUP_CAUSE_DETAIL = {
    pb.HANGUP_CAUSE_UNSPECIFIED: "unspecified",
    pb.HANGUP_CAUSE_NO_ANSWER: "no_answer",
    pb.HANGUP_CAUSE_USER_BUSY: "user_busy",
    pb.HANGUP_CAUSE_NETWORK_OUT_OF_ORDER: "network_out_of_order",
    pb.HANGUP_CAUSE_NORMAL_CLEARING: "normal_clearing",
    pb.HANGUP_CAUSE_CALL_REJECTED: "call_rejected",
}


class _CallState:
    """All per-call state RtcTelephonyClient maintains.

    Created in :meth:`RtcTelephonyClient.dial`, retired in
    :meth:`RtcTelephonyClient._cleanup_call`. Methods called by the
    dispatcher (``on_call_event`` / ``on_dial_ack``) run on the engine
    event loop.
    """

    def __init__(
        self,
        *,
        call_id: int,
        rtc_session: RtcSession,
        edge_uid: str,
        device_id: int,
        outbound_frame_ms: int,
    ) -> None:
        self.call_id = call_id
        self.sid = str(call_id)
        self.rtc_session = rtc_session
        self.edge_uid = edge_uid
        self.device_id = device_id
        self.events_q: asyncio.Queue[TelephonyEvent | object] = asyncio.Queue()
        self.inbound_q: asyncio.Queue[bytes | object] = asyncio.Queue()
        # Forked stream of the SAME inbound PCM frames, so a VAD monitor can
        # run in parallel with ASR without competing for items in inbound_q.
        # ``audio_in()`` consumes ``inbound_q``; ``audio_in_vad()`` consumes
        # ``vad_q``. Both close on ``_SENTINEL``.
        self.vad_q: asyncio.Queue[bytes | object] = asyncio.Queue()
        self.inbound_pump: asyncio.Task[None] | None = None
        self._outbound_ts: int = 0
        self._outbound_frame_ms = outbound_frame_ms
        # Set the first time a terminal event (remote_hangup / device_error /
        # local hangup) closes the iterators, so subsequent terminals
        # don't double-emit sentinels.
        self._closed = False

    # ----- timestamping for push_audio -----------------------------------

    def next_timestamp_ms(self) -> int:
        ts = self._outbound_ts
        self._outbound_ts += self._outbound_frame_ms
        return ts

    # ----- inbound PCM pump ----------------------------------------------

    def start_inbound_pump(self) -> None:
        """Spawn the per-call pump that drains rtc_session.audio_frames()
        into ``self.inbound_q`` (filtered by ``sender_uid``).

        Idempotent — calling twice is a no-op (helps test cleanup
        defensiveness).
        """
        if self.inbound_pump is not None:
            return
        self.inbound_pump = asyncio.create_task(
            self._inbound_loop(),
            name=f"rtc_inbound_pump_{self.sid}",
        )

    async def _inbound_loop(self) -> None:
        # DingRTC C++ SDK mixed-playback OnPlaybackAudioFrame defaults to
        # 48 kHz output regardless of how peers join. ASR Provider (and
        # 8 kHz GSM modem path) need 16 kHz / 8 kHz; resample inline here
        # rather than burdening every consumer. ``audioop.ratecv`` is the
        # stdlib stateful resampler (carries fractional sample state across
        # chunks so a 20 ms 48 kHz chunk → exactly a 20 ms 16 kHz chunk).
        # Python 3.14+ removes audioop; until then this is the lightweight
        # path. ASR ``stream_recognize(audio_chunks)`` is the only consumer
        # in v1.0, so 16 kHz target matches the V3 SAUC required rate.
        try:
            import audioop  # noqa: PLC0415
        except ImportError:  # pragma: no cover  - Py 3.13+ replacement TBD
            audioop = None
        target_rate = 16000
        ratecv_state: Any = None
        try:
            async for frame in self.rtc_session.audio_frames():
                # DingRTC C++ SDK's `OnPlaybackAudioFrame` is **mixed** playback
                # (no per-uid). Binding sets sender_uid = "" for mixed frames.
                # `OnRemoteUserAudioFrame` (per-uid) is not exposed by the
                # Linux 3.9.0 SDK (`expected_remote_uid_` filter exists in
                # audio_observer.cpp but is never triggered on the playback
                # path). In v1.0 a channel has exactly 2 peers (engine +
                # edge), so a mixed frame ≡ edge's audio. Accept empty
                # sender_uid as "from the only other peer = edge"; only
                # skip frames bearing an unexpected non-empty uid (defence
                # in depth against future multi-user channel changes).
                if frame.sender_uid and frame.sender_uid != self.edge_uid:
                    continue
                pcm = frame.pcm
                # Resample 48 kHz → 16 kHz if needed (mixed playback default).
                src_rate = int(frame.sample_rate or 0)
                if (
                    src_rate > 0 and src_rate != target_rate
                    and audioop is not None and pcm
                ):
                    pcm, ratecv_state = audioop.ratecv(
                        pcm, 2, frame.channels or 1,
                        src_rate, target_rate, ratecv_state,
                    )
                await self.inbound_q.put(pcm)
                # Fork the same PCM to the VAD lane. Non-blocking put_nowait
                # would risk dropping frames; await is safe because the VAD
                # monitor drains as fast as ASR.
                await self.vad_q.put(pcm)
        except RtcNotJoined:
            # close_iterators() may race ahead of this task being
            # scheduled: leave() invalidates the session before the pump
            # ever reaches audio_frames(). Treat as a clean exit — the
            # sentinel still goes out in `finally`.
            pass
        finally:
            # Terminator for audio_in() / audio_in_vad() consumers.
            await self.inbound_q.put(_SENTINEL)
            await self.vad_q.put(_SENTINEL)

    # ----- dispatcher-driven event handlers ------------------------------

    async def on_call_event(self, event: pb.CallEvent) -> None:
        kind = event.WhichOneof("kind")
        if kind == "connected":
            await self.events_q.put(
                TelephonyEvent(type="connected", call_id=self.call_id),
            )
        elif kind == "remote_hangup":
            detail = _HANGUP_CAUSE_DETAIL.get(
                event.remote_hangup.cause,
                "unspecified",
            )
            await self.events_q.put(
                TelephonyEvent(
                    type="remote_hangup",
                    call_id=self.call_id,
                    detail=detail,
                ),
            )
            await self.close_iterators()
        elif kind == "device_error":
            await self.events_q.put(
                TelephonyEvent(
                    type="device_error",
                    call_id=self.call_id,
                    detail=event.device_error.code or "",
                ),
            )
            await self.close_iterators()
        elif kind in ("ringing", "user_speaking_started", "user_speaking_stopped"):
            # Informational / A3 markers — no engine state change here.
            return
        else:
            logger.warning(
                "call_event with unknown kind=%s; ignoring",
                kind,
                extra={"call_id": self.sid},
            )

    async def on_dial_ack(self, ack: pb.DialAck) -> None:
        if ack.accepted:
            # Useful for ops monitoring; no engine-side state change.
            return
        # Refused dial — surface as remote_hangup with the edge's reason,
        # then close.
        await self.events_q.put(
            TelephonyEvent(
                type="remote_hangup",
                call_id=self.call_id,
                detail=ack.reason or "dial_rejected",
            ),
        )
        await self.close_iterators()

    # ----- closure -------------------------------------------------------

    async def close_iterators(self) -> None:
        """Emit the sentinel onto ``events_q`` and trigger inbound shutdown
        via :meth:`RtcSession.leave`.

        Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        await self.events_q.put(_SENTINEL)
        # Leaving the channel terminates audio_frames(), which lets the
        # pump's finally block sentinel the inbound queue.
        if self.rtc_session.is_joined:
            with contextlib.suppress(Exception):
                await self.rtc_session.leave()
        if self.inbound_pump is not None:
            # Best-effort wait so the pump's sentinel reaches inbound_q
            # before callers move on. Cancel if the pump misbehaves.
            try:
                await asyncio.wait_for(self.inbound_pump, timeout=2.0)
            except TimeoutError:
                self.inbound_pump.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self.inbound_pump


class RtcTelephonyClient(TelephonyClient):
    """:class:`TelephonyClient` over cloud-edge gRPC + RTC."""

    def __init__(
        self,
        *,
        edge_device_id: str,
        grpc_server: CloudEdgeServer,
        dispatcher: EngineSessionDispatcher,
        token_issuer: RtcTokenIssuer,
        rtc_session_factory: Callable[[], RtcSession],
        outbound_frame_ms: int = 20,
    ) -> None:
        self._edge_device_id = edge_device_id
        self._grpc = grpc_server
        self._dispatcher = dispatcher
        self._issuer = token_issuer
        self._rtc_session_factory = rtc_session_factory
        self._outbound_frame_ms = outbound_frame_ms

        # Engine-side call_id (int) → state.
        self._calls: dict[int, _CallState] = {}

        # Scheduler-injected hints (call_id → device_id), consumed at dial.
        # See :meth:`set_device_for_session`.
        self._pending_devices: dict[int, int] = {}

    # ===== TelephonyClient surface =======================================

    async def dial(self, call_id: int, phone: str) -> None:
        if call_id in self._calls:
            raise RuntimeError(f"call_id already active: {call_id}")

        sid = str(call_id)
        engine_creds, edge_creds = self._issuer.sign_for_call(sid)

        rtc_session = self._rtc_session_factory()
        await rtc_session.join(
            channel=sid,
            token=engine_creds.token,
            uid=engine_creds.user_id,
        )

        state = _CallState(
            call_id=call_id,
            rtc_session=rtc_session,
            edge_uid=edge_creds.user_id,
            device_id=self._pending_devices.pop(call_id, 0),
            outbound_frame_ms=self._outbound_frame_ms,
        )
        self._calls[call_id] = state
        state.start_inbound_pump()

        # Bind dispatcher BEFORE sending DialCommand — otherwise a fast
        # DialAck / CallEvent could land in the dispatcher before there's
        # a session to deliver it to.
        self._dispatcher.register(
            sid,
            edge_device_id=self._edge_device_id,
            on_call_event=state.on_call_event,
            on_dial_ack=state.on_dial_ack,
        )

        dial_cmd = pb.DialCommand(
            call_id=sid,
            device_id=state.device_id,
            number=phone,
            caller_id="",  # populated by the scheduler in a future PR
            rtc_channel=sid,
            rtc_token=edge_creds.token,
            rtc_uid_edge=edge_creds.user_id,
            rtc_uid_engine=engine_creds.user_id,
        )
        try:
            await self._grpc.send_to_edge(
                self._edge_device_id,
                pb.Cloud2Edge(dial=dial_cmd),
            )
        except EdgeNotConnected:
            # Edge isn't connected — surface as device_error and unwind.
            await state.events_q.put(
                TelephonyEvent(
                    type="device_error",
                    call_id=call_id,
                    detail="edge_disconnected",
                ),
            )
            await state.close_iterators()
            await self._purge_call(call_id)
            raise

    async def hangup(self, call_id: int) -> None:
        state = self._calls.get(call_id)
        if state is None:
            return

        # Best-effort cancel toward the edge — the edge may already be
        # gone (call ended remotely a tick ago), which is fine.
        with contextlib.suppress(EdgeNotConnected):
            await self._grpc.send_to_edge(
                self._edge_device_id,
                pb.Cloud2Edge(
                    cancel=pb.CancelCommand(call_id=state.sid, reason="local hangup"),
                ),
            )

        # Synthesise the local_hangup event ahead of the sentinel so
        # consumers see "local_hangup" before the iterator terminates,
        # matching RealTelephonyClient's contract.
        await state.events_q.put(
            TelephonyEvent(type="local_hangup", call_id=call_id),
        )
        await state.close_iterators()
        await self._purge_call(call_id)

    def audio_in(self, call_id: int) -> AsyncIterator[bytes]:
        state = self._require_state(call_id)

        async def _iter() -> AsyncIterator[bytes]:
            while True:
                item = await state.inbound_q.get()
                if item is _SENTINEL:
                    return
                assert isinstance(item, bytes)
                yield item

        return _iter()

    def audio_in_vad(self, call_id: int) -> AsyncIterator[bytes]:
        """Fork of the inbound PCM stream for a VAD monitor.

        Same frames as ``audio_in()`` but on a parallel queue so VAD-based
        barge-in detection can run alongside ASR without ASR-vendor partial
        latency. Closes on the same ``_SENTINEL`` as ``audio_in``.
        """
        state = self._require_state(call_id)

        async def _iter() -> AsyncIterator[bytes]:
            while True:
                item = await state.vad_q.get()
                if item is _SENTINEL:
                    return
                assert isinstance(item, bytes)
                yield item

        return _iter()

    async def audio_out(self, call_id: int, chunks: AsyncIterator[bytes]) -> None:
        state = self._require_state(call_id)
        async for chunk in chunks:
            await state.rtc_session.push_audio(
                chunk,
                timestamp_ms=state.next_timestamp_ms(),
            )

    def events(self, call_id: int) -> AsyncIterator[TelephonyEvent]:
        state = self._require_state(call_id)

        async def _iter() -> AsyncIterator[TelephonyEvent]:
            while True:
                item = await state.events_q.get()
                if item is _SENTINEL:
                    return
                assert isinstance(item, TelephonyEvent)
                yield item

        return _iter()

    # ===== Hints / internals =============================================

    def set_device_for_session(self, call_id: int, device_id: int) -> None:
        """Stash a device_id to be carried in the next ``dial(call_id, ...)``.

        Same contract as :meth:`RealTelephonyClient.set_device_for_session`:
        the scheduler picks the device first, then calls this before
        :meth:`dial`. Multiple hints overwrite; the hint is consumed (popped)
        at dial time.
        """
        self._pending_devices[call_id] = device_id

    def _require_state(self, call_id: int) -> _CallState:
        state = self._calls.get(call_id)
        if state is None:
            raise RuntimeError(f"unknown call_id: {call_id}")
        return state

    async def _purge_call(self, call_id: int) -> None:
        """Drop call_id from the live registry. Idempotent."""
        state = self._calls.pop(call_id, None)
        if state is None:
            return
        self._dispatcher.deregister(state.sid)


__all__ = ["RtcTelephonyClient"]
