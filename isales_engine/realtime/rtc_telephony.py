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
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from isales_common.audio.rtc import RtcNotJoined, RtcSession
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import (
    CloudEdgeServer,
    EdgeNotConnected,
)

from isales_engine.realtime.ambient import (
    FRAME_BYTES,
    SILENCE_FRAME,
    AmbientReader,
    apply_fadeout,
    get_ambient_loop,
    mix_frame,
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

# Outbound playout jitter cushion (ambient-background-mix follow-up). The
# mixing pump drains the playout queue at a strict 10 ms wall-clock pace and
# substitutes a background-only SILENCE frame whenever the queue is empty. With
# no cushion, any transient TTS underflow (bursty / slower-than-realtime vendor
# delivery, worst on a turn's first sentence) is punched into the speech as a
# silence gap → choppy / 生硬. Holding ~200 ms before releasing a burst to the
# pump gives it a buffer that absorbs the stall. 20 frames × 10 ms = 200 ms.
PLAYOUT_PREBUFFER_FRAMES = 20

# engine-barge-in-fade-out: default fade-out window applied to the still-buffered
# TTS when a barge-in cancels the AI mid-utterance, instead of a hard cut to
# silence. ~100 ms reads as a natural human "trail-off" while staying well inside
# the ~200 ms turn-gap floor; 15-30 ms would only declick. 0 = keep the old hard
# cut. The per-campaign value (campaign.barge_in_fadeout_ms) overrides this via
# set_barge_in_fadeout(); this constant is the default when none is set.
DEFAULT_BARGE_IN_FADEOUT_MS = 100


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
        edge_device_id: str,
    ) -> None:
        self.call_id = call_id
        self.sid = str(call_id)
        self.rtc_session = rtc_session
        self.edge_uid = edge_uid
        self.device_id = device_id
        self.edge_device_id = edge_device_id
        self.events_q: asyncio.Queue[TelephonyEvent | object] = asyncio.Queue()
        self.inbound_q: asyncio.Queue[bytes | object] = asyncio.Queue()
        # Forked stream of the SAME inbound PCM frames, so a VAD monitor can
        # run in parallel with ASR without competing for items in inbound_q.
        # ``audio_in()`` consumes ``inbound_q``; ``audio_in_vad()`` consumes
        # ``vad_q``. Both close on ``_SENTINEL``.
        self.vad_q: asyncio.Queue[bytes | object] = asyncio.Queue()
        self.inbound_pump: asyncio.Task[None] | None = None
        # Set the first time a terminal event (remote_hangup / device_error /
        # local hangup) closes the iterators, so subsequent terminals
        # don't double-emit sentinels.
        self._closed = False

        # engine-barge-in-fade-out: outbound TTS ALWAYS flows through one queue +
        # mixing pump (start_outbound_pump), regardless of ambient — so a barge-in
        # has buffered audio to fade out in exactly one place (_flush_playout) and
        # the old no-buffer direct-push path is gone. ``ambient_active`` now only
        # selects whether a background bed is mixed in and whether the ~200 ms
        # jitter cushion applies (it does not for the bare-TTS case, to preserve
        # first-audio latency). ``_barge_in_fadeout_ms`` is the fade window
        # applied on cancel (DEFAULT_BARGE_IN_FADEOUT_MS until a per-campaign
        # value is wired in).
        self.ambient_active = False
        self.ambient_reader: AmbientReader | None = None
        self.ambient_gain = 0.0
        self.playout_q: asyncio.Queue[bytes] | None = None
        self.outbound_pump: asyncio.Task[None] | None = None
        self._pump_ts = 0
        self._barge_in_fadeout_ms = DEFAULT_BARGE_IN_FADEOUT_MS

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
        import time as _time  # noqa: PLC0415  # DIAG-REMOVE-AFTER-MIC-DEBUG
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
        # asr-noise-gate (experiment 2026-06-05) ----------------------------
        # 与 asr_volcengine._push_audio 的 100ms 攒包配套使用。喂 ASR 那一路里,
        # 低于阈值且已连续静音超过 hangover 的帧, 替换成数字静音 (全 0 PCM),
        # 让 vendor 在规格正确的包下按 end_window_size(400ms) 及时切句。
        # VAD 那一路 (self.vad_q) 保持原始音频, 不门控 (barge-in 检测要原信号)。
        # 阈值/hangover 走 env; gate_rms=0 关闭。
        # Removal/promote trigger: 真机验证空档塌到 <1s 且不碎句/不削轻声用户 →
        # 固化为 campaign 配置 + 落 openspec change; 若碎句/削轻声, 回退本块。
        import os as _os  # noqa: PLC0415
        _gate_rms = int(_os.environ.get("ISALES_ASR_NOISE_GATE_RMS", "1500"))
        _gate_hangover_ms = float(_os.environ.get("ISALES_ASR_NOISE_GATE_HANGOVER_MS", "300"))
        _gate_silence_ms = 0.0
        _gate_active = False
        # asr-noise-gate end -------------------------------------------------
        # DIAG-REMOVE-AFTER-MIC-DEBUG begin ----------------------------------
        recv_count = 0
        last_log_t = _time.monotonic()
        last_log_count = 0
        max_rms_window = 0
        first_sender_uid: str | None = None
        skipped_sender_count = 0
        # DIAG: raw stereo 48k dump of the first 15 s of inbound, for
        # measuring the real inbound signal (per-channel RMS over time).
        # Controlled test: user stays SILENT during the greeting, then afplay /
        # measure the dump — non-silent inbound while only the AI is talking
        # would indicate the engine's own TTS reaching inbound; pure silence
        # rules that out. (Use this to ground the VAD-threshold re-evaluation;
        # the earlier "self-loopback" conclusion is retracted, pending this
        # measurement.)
        _raw_dump_path = "/tmp/inbound_raw_48k_stereo.pcm"
        _raw_dump_fh: Any = None
        _raw_dump_max_bytes = 15 * 48000 * 2 * 2  # 15s @ 48k stereo int16
        _raw_dump_written = 0
        try:
            _raw_dump_fh = open(_raw_dump_path, "wb")
        except Exception:  # noqa: BLE001
            pass
        # DIAG-REMOVE-AFTER-MIC-DEBUG end ------------------------------------
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
                    skipped_sender_count += 1  # DIAG-REMOVE-AFTER-MIC-DEBUG
                    continue
                # DIAG-REMOVE-AFTER-ECHO-TEST: dump raw stereo 48k frame.pcm
                if _raw_dump_fh and _raw_dump_written < _raw_dump_max_bytes:
                    _raw_dump_fh.write(frame.pcm)
                    _raw_dump_written += len(frame.pcm)
                    if _raw_dump_written >= _raw_dump_max_bytes:
                        _raw_dump_fh.close()
                        _raw_dump_fh = None
                        logger.info(
                            "inbound_raw_dump_complete path=%s bytes=%s "
                            "(15s @ 48k stereo int16)",
                            _raw_dump_path, _raw_dump_written,
                        )
                # DIAG-REMOVE-AFTER-MIC-DEBUG begin ---------------------------
                if first_sender_uid is None:
                    first_sender_uid = frame.sender_uid or "<empty>"
                recv_count += 1
                _p = frame.pcm
                if _p and len(_p) >= 2:
                    _sample_count = len(_p) // 2
                    _stride = max(1, _sample_count // 32)
                    _total = 0
                    _seen = 0
                    for _i in range(0, len(_p), _stride * 2):
                        if _i + 2 > len(_p):
                            break
                        _s = int.from_bytes(_p[_i:_i + 2], "little", signed=True)
                        _total += _s * _s
                        _seen += 1
                    if _seen:
                        _rms = int((_total // _seen) ** 0.5)
                        if _rms > max_rms_window:
                            max_rms_window = _rms
                # DIAG-REMOVE-AFTER-MIC-DEBUG end -----------------------------
                pcm = frame.pcm
                # DingRTC mixed playback frame on Linux 3.9 SDK is 48 kHz
                # **stereo** int16 10 ms (1920 B = 480 samples × 2 ch × 2 B).
                # ASR / VAD downstream consume 16 kHz **mono** int16. Two-step
                # normalize: (1) stereo → mono (L+R)/2 with audioop.tomono,
                # (2) 48 kHz → 16 kHz with audioop.ratecv using channels=1
                # (post-downmix). Without (1) ratecv preserves stereo and the
                # downstream ASR provider sees 16 kHz stereo bytes interpreted
                # as 16 kHz mono → effective 2× tempo → vendor V3 SAUC returns
                # `audio_info.duration=0 text=""` empty results (2026-06-01
                # diagnostic on mac dev-no-modem fake-WAV upstream).
                src_rate = int(frame.sample_rate or 0)
                src_channels = int(getattr(frame, "channels", 1) or 1)
                if src_channels > 1 and audioop is not None and pcm:
                    pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
                # DIAG-REMOVE-AFTER-MIC-DEBUG: post-downmix RMS — verify
                # stereo→mono didn't cancel L+R (phase-inverted channels).
                _post_downmix_rms = 0
                if audioop is not None and pcm:
                    try:
                        _post_downmix_rms = audioop.rms(pcm, 2)
                    except Exception:  # noqa: BLE001
                        pass
                if (
                    src_rate > 0 and src_rate != target_rate
                    and audioop is not None and pcm
                ):
                    pcm, ratecv_state = audioop.ratecv(
                        pcm, 2, 1,
                        src_rate, target_rate, ratecv_state,
                    )
                # DIAG-REMOVE-AFTER-MIC-DEBUG: post-resample RMS (final PCM going to ASR)
                _final_rms = 0
                if audioop is not None and pcm:
                    try:
                        _final_rms = audioop.rms(pcm, 2)
                    except Exception:  # noqa: BLE001
                        pass
                # asr-noise-gate: 见 _inbound_loop 顶部说明。低于阈值且已连续
                # 静音超过 hangover → 喂 vendor 数字静音, 触发 end_window 及时切句。
                _asr_pcm = pcm
                if _gate_rms > 0 and pcm:
                    _frame_ms = (len(pcm) / 2.0) / (target_rate / 1000.0)
                    if _final_rms >= _gate_rms:
                        _gate_silence_ms = 0.0
                    else:
                        _gate_silence_ms += _frame_ms
                        if _gate_silence_ms > _gate_hangover_ms:
                            _asr_pcm = b"\x00" * len(pcm)
                    _now_gating = _asr_pcm is not pcm
                    if _now_gating != _gate_active:
                        _gate_active = _now_gating
                        logger.info(
                            "asr_noise_gate sid=%s state=%s final_rms=%s "
                            "silence_ms=%.0f gate_rms=%s",
                            self.sid, "ENGAGED" if _now_gating else "RELEASED",
                            _final_rms, _gate_silence_ms, _gate_rms,
                        )
                await self.inbound_q.put(_asr_pcm)
                # Fork the same PCM to the VAD lane. Non-blocking put_nowait
                # would risk dropping frames; await is safe because the VAD
                # monitor drains as fast as ASR.
                await self.vad_q.put(pcm)
                # DIAG-REMOVE-AFTER-MIC-DEBUG begin ---------------------------
                _now = _time.monotonic()
                if _now - last_log_t >= 1.0:
                    logger.info(
                        "rtc_inbound_1s sid=%s recv=%s delta=%s raw_max_rms=%s "
                        "post_downmix_rms=%s final_rms=%s "
                        "sample_rate=%s channels=%s bytes=%s first_sender_uid=%s "
                        "skipped_sender=%s edge_uid=%s after_resample_bytes=%s",
                        self.sid, recv_count, recv_count - last_log_count,
                        max_rms_window, _post_downmix_rms, _final_rms,
                        frame.sample_rate,
                        getattr(frame, "channels", "?"),
                        len(_p),
                        first_sender_uid, skipped_sender_count, self.edge_uid,
                        len(pcm),
                    )
                    last_log_t = _now
                    last_log_count = recv_count
                    max_rms_window = 0
                # DIAG-REMOVE-AFTER-MIC-DEBUG end -----------------------------
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

    # ----- outbound mixing pump ------------------------------------------

    def start_outbound_pump(self) -> None:
        """Start the always-on outbound playout buffer + mixing pump.

        engine-barge-in-fade-out D3: ALL outbound TTS flows through this queue +
        pump whether or not a background bed is configured, so the barge-in
        fade-out (_flush_playout) has buffered audio to ramp down in one place.
        Idempotent — once started the pump is per-call. The pump reads
        ``ambient_reader`` / ``ambient_gain`` per frame, so ``enable_ambient``
        may flip the background on AFTER the pump is already running.
        """
        if self.outbound_pump is not None:
            return
        # ~3 s of buffered TTS frames before back-pressuring audio_out. TTS
        # arrives faster than real-time (a single chunk can be 250-400 ms);
        # the pump drains at exactly real-time, so a bound is needed.
        self.playout_q = asyncio.Queue(maxsize=300)
        self.outbound_pump = asyncio.create_task(
            self._outbound_loop(), name=f"rtc_outbound_pump_{self.sid}",
        )

    def enable_ambient(self, reader: AmbientReader, gain: float) -> None:
        """Mix a continuous background bed into the outbound stream.

        Only sets the background content + gain; the outbound pump itself is
        started unconditionally by :meth:`start_outbound_pump`. Idempotent — a
        second call is a no-op. ``ambient-background-mix`` D1/D2.
        """
        if self.ambient_active:
            return
        self.start_outbound_pump()
        self.ambient_reader = reader
        self.ambient_gain = gain
        self.ambient_active = True

    async def feed_playout(self, chunks: AsyncIterator[bytes]) -> None:
        """Slice a TTS PCM stream into 10 ms frames and enqueue for the pump.

        Returns once the pump has consumed (pushed) every enqueued frame, so the
        caller's "playback complete" contract still holds. On ``CancelledError``
        (barge-in) the still-buffered TTS is faded out rather than hard-cut
        (engine-barge-in-fade-out) — the pump keeps running with background only.
        ``ambient-background-mix`` D2/D5.
        """
        assert self.playout_q is not None
        q = self.playout_q
        leftover = b""
        # Jitter cushion: hold the first PLAYOUT_PREBUFFER_FRAMES (~200 ms)
        # before releasing any frame to the pump, so each burst starts with a
        # buffer that absorbs a transient TTS underflow instead of letting the
        # pump punch a silence gap into the speech. ONLY applied when a
        # background bed is mixed (ambient_active): the bed makes the pump's
        # strict 10 ms grid audible across underflows, so smoothing earns its
        # latency. For bare TTS (no ambient — the default) the cushion is
        # skipped (primed from the start) so first-audio latency matches the old
        # direct-push path; the natural queue backlog (TTS bursts ahead of
        # real-time) still supplies frames for the barge-in fade-out.
        # engine-barge-in-fade-out R1.
        cushion: list[bytes] = []
        primed = not self.ambient_active
        try:
            async for chunk in chunks:
                buf = leftover + chunk
                off = 0
                while off + FRAME_BYTES <= len(buf):
                    frame = buf[off:off + FRAME_BYTES]
                    off += FRAME_BYTES
                    if primed:
                        await q.put(frame)
                    else:
                        cushion.append(frame)
                        if len(cushion) >= PLAYOUT_PREBUFFER_FRAMES:
                            for f in cushion:
                                await q.put(f)
                            cushion.clear()
                            primed = True
                leftover = buf[off:]
            # Stream ended: release any held cushion (sentence shorter than the
            # cushion), then pad + enqueue the final partial frame.
            for f in cushion:
                await q.put(f)
            if leftover:  # pad the final partial frame to a full 10 ms shape
                await q.put(leftover + b"\x00" * (FRAME_BYTES - len(leftover)))
            await q.join()
        except asyncio.CancelledError:
            self._flush_playout(self._barge_in_fadeout_ms)
            raise

    def _flush_playout(self, fadeout_ms: int = 0) -> None:
        """Stop buffered TTS playback on barge-in.

        engine-barge-in-fade-out: drain everything still queued, then — when
        ``fadeout_ms > 0`` — re-queue the first ``ceil(fadeout_ms/10)`` frames
        ramped down to silence so the AI voice trails off instead of clicking to
        zero. ``fadeout_ms == 0`` keeps the legacy hard cut (drop everything).
        Synchronous, so the pump cannot interleave between the drain and the
        re-queue. ``task_done`` accounting stays balanced: every drained frame is
        marked done here; each re-queued faded frame is a fresh item the pump
        marks done when it pushes it.
        """
        q = self.playout_q
        if q is None:
            return
        pending: list[bytes] = []
        while True:
            try:
                pending.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
            q.task_done()
        if fadeout_ms <= 0 or not pending:
            return  # legacy hard cut, or nothing buffered to fade
        n = min(-(-fadeout_ms // 10), len(pending))  # ceil(fadeout_ms/10), capped
        for frame in apply_fadeout(pending[:n]):
            q.put_nowait(frame)

    async def _outbound_loop(self) -> None:
        """Push one mixed 10 ms frame every 10 ms of wall-clock.

        Monotonic absolute scheduling (target = start + idx·10ms) so jitter
        doesn't accumulate drift — a frame that fires late is followed
        immediately, not 10 ms later. ``ambient-background-mix`` D1/D4.

        ``ambient_reader`` / ``ambient_gain`` are read PER FRAME (not captured at
        task start) so a background bed enabled after the pump is already running
        (engine-barge-in-fade-out always starts the pump up front) still takes
        effect. With no bed (``reader is None`` / ``gain == 0``) ``mix_frame``
        returns the TTS frame unchanged, i.e. bare-TTS playout.
        """
        q = self.playout_q
        frame_dt = 0.01
        start = time.monotonic()
        idx = 0
        try:
            while True:
                idx += 1
                target = start + idx * frame_dt
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if q is not None:
                    try:
                        tts = q.get_nowait()
                        has_tts = True
                    except asyncio.QueueEmpty:
                        tts = SILENCE_FRAME
                        has_tts = False
                else:  # pragma: no cover - start_outbound_pump always sets q
                    tts = SILENCE_FRAME
                    has_tts = False
                reader = self.ambient_reader
                bg = reader.next_frame() if reader is not None else SILENCE_FRAME
                frame = mix_frame(tts, bg, self.ambient_gain)
                try:
                    await self.rtc_session.push_audio(frame, timestamp_ms=self._pump_ts)
                except RtcNotJoined:
                    if has_tts and q is not None:
                        q.task_done()
                    break
                except Exception:  # noqa: BLE001 - one bad frame must not kill the pump
                    logger.warning("ambient_pump_push_failed sid=%s", self.sid, exc_info=True)
                else:
                    self._pump_ts += 10
                if has_tts and q is not None:
                    q.task_done()
        except asyncio.CancelledError:
            raise

    async def stop_outbound_pump(self) -> None:
        pump = self.outbound_pump
        if pump is None:
            return
        self.outbound_pump = None
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump

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
        # Stop the outbound mixing pump first so it stops pushing before the
        # channel leaves (avoids RtcNotJoined spam from in-flight frames).
        await self.stop_outbound_pump()
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
    ) -> None:
        self._legacy_edge_device_id = edge_device_id
        self._grpc = grpc_server
        self._dispatcher = dispatcher
        self._issuer = token_issuer
        self._rtc_session_factory = rtc_session_factory

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
            edge_device_id="",  # resolved below
        )

        # Resolve the target edge: if a legacy edge_device_id was configured,
        # always route to that single edge; otherwise use the dynamic
        # device→edge mapping maintained by the gRPC server from heartbeats.
        target_edge: str
        if self._legacy_edge_device_id:
            target_edge = self._legacy_edge_device_id
        else:
            target_edge = self._grpc.resolve_edge_for_device(state.device_id)
        state.edge_device_id = target_edge

        self._calls[call_id] = state
        state.start_inbound_pump()

        # Bind dispatcher BEFORE sending DialCommand — otherwise a fast
        # DialAck / CallEvent could land in the dispatcher before there's
        # a session to deliver it to.
        self._dispatcher.register(
            sid,
            edge_device_id=target_edge,
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
                target_edge,
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
                state.edge_device_id,
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
        # engine-barge-in-fade-out D3: unified outbound path — always feed the
        # pump (started up front by configure_ambient; this guards any caller
        # that reaches audio_out first). The pump mixes a background bed only
        # when one is configured, otherwise pushes the TTS frames unchanged.
        state.start_outbound_pump()
        await state.feed_playout(chunks)

    def configure_ambient(
        self, call_id: int, asset: str | None, gain: float
    ) -> None:
        """Start the outbound pump and (optionally) enable background mixing.

        Called by run_loop after the call connects, before the greeting. The
        outbound mixing pump is started unconditionally (engine-barge-in-fade-out
        D3) so barge-in fade-out works whether or not a bed is configured. Empty
        ``asset`` → pump on, no background. A missing / unreadable asset logs and
        proceeds without background rather than failing. ``ambient-background-mix``
        D6.
        """
        state = self._calls.get(call_id)
        if state is None:
            return
        state.start_outbound_pump()
        if not asset:
            return
        loop = get_ambient_loop(asset)
        if loop is None:
            logger.warning(
                "ambient_disabled call=%s asset=%s reason=unavailable",
                call_id, asset,
            )
            return
        state.enable_ambient(loop.reader(), gain)
        logger.info("ambient_enabled call=%s asset=%s gain=%.3f", call_id, asset, gain)

    def set_barge_in_fadeout(self, call_id: int, fadeout_ms: int) -> None:
        """Set the per-call barge-in fade-out window. engine-barge-in-fade-out."""
        state = self._calls.get(call_id)
        if state is None:
            return
        state._barge_in_fadeout_ms = fadeout_ms

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
