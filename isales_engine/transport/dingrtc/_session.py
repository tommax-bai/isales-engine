"""DingRtcSession — RtcSession ABC impl + production channel adapter.

Spec: openspec/changes/engine-rtc-dingrtc-migration/specs/device-hardware/
      § Requirement: 云端 isales-engine 通过项目内 pybind11 binding 接入
      DingRTC 3.x Linux C++ SDK.

DingRTC SDK threading model is simpler than the legacy ARTC sidecar:

- C++ SDK runs its own internal worker threads (audio I/O, network).
- ``RegisterAudioFrameObserver`` callbacks fire on those threads → the C++
  binding's ``AudioObserver`` copies the frame bytes into a thread-safe
  ``FrameRingBuffer``, returning immediately.
- ``RegisterEngineEventListener`` callbacks (join / leave / error) fire on
  SDK threads → the binding's ``EngineListener`` invokes Python callables
  with the GIL acquired but on the SDK thread.
- :class:`DingRtcSession` registers Python callbacks that ``call_soon_thread
  safe`` onto the engine's asyncio loop, where the business code lives.

No driver thread / pump loop required — that was the legacy ARTC SDK
sidecar IPC workaround, which DingRTC eliminates by being in-process C++.

Production wiring is in :class:`_PybindDingRtcChannel` below. Tests use
:class:`InMemoryDingRtcChannel` (from ``_in_memory.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, Self

from isales_common.audio.rtc import (
    PcmFrame,
    RtcError,
    RtcNotJoined,
    RtcPushBackpressure,
    RtcSession,
)

__all__ = [
    "DingRtcChannelProtocol",
    "DingRtcSession",
    "InboundFrameCallback",
    "JoinResultCallback",
    "ErrorCallback",
    "SdkLoadError",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Callback signatures
# =============================================================================

#: Called once per inbound audio frame (sender_uid, pcm, sample_rate, timestamp_ms).
#: sender_uid is empty for OnPlaybackAudioFrame (混流), set for OnRemoteUserAudioFrame.
InboundFrameCallback = Callable[[str, bytes, int, int], None]

#: Called when SDK reports the result of JoinChannel: (errCode, channel, userId, elapsed_ms).
JoinResultCallback = Callable[[int, str, str, int], None]

#: Called when SDK signals an error: (errCode, message).
ErrorCallback = Callable[[int, str], None]


# =============================================================================
# SDK load error (mirror of legacy SdkLoadError for backward compat)
# =============================================================================


class SdkLoadError(RtcError):
    """Raised when the ``dingrtc_pywrap`` extension module cannot be loaded.

    Surfaced from :meth:`DingRtcSession.production` (which lazy-imports
    the binding). Tests avoid this entirely by injecting an
    :class:`InMemoryDingRtcChannel`.
    """


# =============================================================================
# Channel protocol
# =============================================================================


class DingRtcChannelProtocol(Protocol):
    """Narrow surface the session needs from any backing channel.

    The production impl (:class:`_PybindDingRtcChannel`) wraps the C++
    pybind11 binding; the test impl (:class:`InMemoryDingRtcChannel`)
    operates fully in-memory. Both implement these async methods + the
    three callback setters.
    """

    async def setup_and_join(
        self,
        *,
        channel: str,
        token: str,
        uid: str,
        app_id: str,
        send_sample_rate: int,
        send_channels: int,
        gslb_server: str,
        join_timeout: float,
    ) -> None:
        """Configure SDK + JoinChannel + await OnJoinChannelResult.

        Raises :class:`RtcError` with the SDK error code if join fails
        (e.g. ``RtcEngineErrorJoinBadToken = 0x02010205``) or times out.
        """

    async def leave(self) -> None:
        """Tear down SDK resources. Idempotent."""

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        """Push outbound PCM. Returns SDK rc (0 = ok, non-zero = error).

        For PushExternalAudioFrame buffer-full errors the impl raises
        :class:`RtcPushBackpressure`; other non-zero rcs raise
        :class:`RtcError`.
        """

    def drain_inbound_frames(self, max_n: int = 8) -> list[Any]:
        """Drain up to ``max_n`` accumulated inbound frames.

        Returns a list of objects with ``pcm`` / ``sample_rate`` /
        ``channels`` / ``remote_uid`` / ``timestamp`` attrs (matching the
        pybind11 ``PcmFrame`` C++ struct).
        """

    def set_on_join_result(self, cb: JoinResultCallback | None) -> None: ...
    def set_on_error(self, cb: ErrorCallback | None) -> None: ...


# =============================================================================
# Session
# =============================================================================


class DingRtcSession(RtcSession):
    """RtcSession backed by a :class:`DingRtcChannelProtocol`.

    Lifecycle:

        sess = DingRtcSession.production(app_id="o6dpsan9...")
        await sess.join(
            channel="call-123",
            token=mint_dingrtc_token(...),
            uid="engine-call-123",
        )
        async for frame in sess.audio_frames():
            feed_to_asr(frame.pcm)
        await sess.push_audio(tts_chunk, timestamp_ms=now_ms())
        await sess.leave()

    Construction modes:

    - :meth:`production` — lazy-imports the ``dingrtc_pywrap`` extension and
      wires a :class:`_PybindDingRtcChannel`. Raises :class:`SdkLoadError`
      if the binding is unavailable.
    - ``DingRtcSession(channel=InMemoryDingRtcChannel(...), app_id="...")``
      — tests inject a channel directly.
    """

    def __init__(
        self,
        *,
        channel: DingRtcChannelProtocol,
        app_id: str,
        inbound_buffer_size: int = 1024,
        gslb_server: str = "https://gslb.dingrtc.com",
        join_timeout: float = 10.0,
        drain_interval_ms: float = 20.0,
    ) -> None:
        self._channel = channel
        self._app_id = app_id
        self._gslb_server = gslb_server
        self._join_timeout = join_timeout
        self._drain_interval_s = drain_interval_ms / 1000.0
        self._inbound: asyncio.Queue[PcmFrame | None] = asyncio.Queue(
            maxsize=inbound_buffer_size,
        )
        self._joined = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drainer_task: asyncio.Task[None] | None = None
        self._send_sample_rate = 16000
        self._send_channels = 1

    # ----- factory --------------------------------------------------------

    @classmethod
    def production(cls, *, app_id: str, **kwargs: Any) -> Self:
        """Build a session wired to the real DingRTC SDK.

        Raises :class:`SdkLoadError` if ``dingrtc_pywrap`` cannot be
        imported (binding not built / SDK shared lib not on LD_LIBRARY_PATH).

        Tests SHOULD construct ``cls(channel=InMemoryDingRtcChannel(),
        app_id=...)`` directly to avoid the import.
        """
        try:
            import dingrtc_pywrap  # noqa: PLC0415
        except ImportError as exc:
            raise SdkLoadError(
                "dingrtc_pywrap extension not loadable; run "
                "deploy/cloud/scripts/build-dingrtc-binding.sh and ensure "
                "DingRTC SDK is at $ISALES_DINGRTC_LINUX_SDK_PATH "
                "(default /opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/)"
            ) from exc
        channel = _PybindDingRtcChannel(binding=dingrtc_pywrap)
        return cls(channel=channel, app_id=app_id, **kwargs)

    # ----- lifecycle ------------------------------------------------------

    async def join(
        self,
        channel: str,
        token: str,
        uid: str,
        *,
        send_sample_rate: int = 16000,
        send_channels: int = 1,
    ) -> None:
        if self._joined:
            raise RtcError("session already joined")

        self._loop = asyncio.get_running_loop()
        self._send_sample_rate = send_sample_rate
        self._send_channels = send_channels

        # Wire the join result + error callbacks BEFORE asking the channel
        # to join — once setup_and_join schedules the SDK JoinChannel call,
        # the callback may fire from an SDK thread before setup_and_join's
        # await even returns.
        self._channel.set_on_join_result(self._on_join_result_threadsafe)
        self._channel.set_on_error(self._on_error_threadsafe)

        await self._channel.setup_and_join(
            channel=channel,
            token=token,
            uid=uid,
            app_id=self._app_id,
            send_sample_rate=send_sample_rate,
            send_channels=send_channels,
            gslb_server=self._gslb_server,
            join_timeout=self._join_timeout,
        )
        self._joined = True

        # Start the inbound drainer task (no separate thread; just a
        # periodic asyncio task that polls the C++ ring buffer and
        # forwards frames to the inbound queue).
        self._drainer_task = asyncio.create_task(
            self._drain_inbound_loop(),
            name=f"dingrtc-drainer-{uid}",
        )

    async def leave(self) -> None:
        if not self._joined:
            return
        self._joined = False

        # Stop drainer before tearing down channel — otherwise it might
        # call into a destroyed engine.
        if self._drainer_task is not None:
            self._drainer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drainer_task
            self._drainer_task = None

        await self._channel.leave()

        # Clear callbacks so any late SDK-thread invocation is a no-op.
        self._channel.set_on_join_result(None)
        self._channel.set_on_error(None)

        # Sentinel ends any in-flight audio_frames() iterator.
        await self._inbound.put(None)

    @property
    def is_joined(self) -> bool:
        return self._joined

    # ----- inbound --------------------------------------------------------

    async def audio_frames(self) -> AsyncIterator[PcmFrame]:
        if not self._joined:
            raise RtcNotJoined("audio_frames() called before join()")
        while True:
            frame = await self._inbound.get()
            if frame is None:
                return
            yield frame

    async def _drain_inbound_loop(self) -> None:
        """Periodically pull frames from the binding's ring buffer and
        push them to the inbound asyncio.Queue.

        At ~50 Hz SDK frame rate + 20 ms drain interval, each drain pulls
        ~1 frame. We drain up to 8 at a time as a small batch (sized
        relative to the 64-frame ring buffer capacity); if the producer
        burst-fills, the C++ buffer overwrite-oldest semantics protect
        the SDK from blocking.
        """
        try:
            while self._joined:
                frames = self._channel.drain_inbound_frames(max_n=8)
                for raw in frames:
                    sender_uid = getattr(raw, "remote_uid", "") or ""
                    pcm_bytes = bytes(getattr(raw, "pcm", b""))
                    sample_rate = int(getattr(raw, "sample_rate", 0) or 0)
                    channels = int(getattr(raw, "channels", 1) or 1)
                    timestamp = int(getattr(raw, "timestamp", 0) or 0)
                    frame = PcmFrame(
                        sender_uid=sender_uid,
                        pcm=pcm_bytes,
                        sample_rate=sample_rate,
                        channels=channels,
                        timestamp_ms=timestamp,
                    )
                    with contextlib.suppress(asyncio.QueueFull):
                        self._inbound.put_nowait(frame)
                await asyncio.sleep(self._drain_interval_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dingrtc drainer task crashed; session will hang on audio_frames()")
            raise

    # ----- outbound -------------------------------------------------------

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> None:
        if not self._joined:
            raise RtcNotJoined("push_audio() called before join()")
        rc = await self._channel.push_audio(pcm, timestamp_ms=timestamp_ms)
        if rc != 0:
            # Channel impl typically raises RtcPushBackpressure / RtcError
            # before reaching here; this is a fallback for impls that
            # return non-zero rcs without raising.
            raise RtcError(f"push_audio rc={rc} (0x{rc & 0xFFFFFFFF:08X})")

    # ----- callback marshaling -------------------------------------------

    def _on_join_result_threadsafe(
        self,
        result: int,
        channel: str,
        user_id: str,
        elapsed: int,
    ) -> None:
        """Called from SDK callback thread by the channel impl.

        Note: the channel impl typically resolves a join-future
        internally and re-surfaces the result through ``setup_and_join``.
        This hook is for any additional bookkeeping (e.g. metrics).
        """
        logger.info(
            "DingRTC join result: rc=%d (0x%08X) channel=%s uid=%s elapsed_ms=%d",
            result, result & 0xFFFFFFFF, channel, user_id, elapsed,
        )

    def _on_error_threadsafe(self, code: int, message: str) -> None:
        """Called from SDK callback thread by the channel impl."""
        logger.warning(
            "DingRTC error: code=%d (0x%08X) message=%r",
            code, code & 0xFFFFFFFF, message,
        )


# =============================================================================
# Production channel impl (pybind11)
# =============================================================================


class _PybindDingRtcChannel:
    """Production :class:`DingRtcChannelProtocol` impl backed by dingrtc_pywrap.

    Lifecycle convention:

    1. ``setup_and_join`` creates a fresh EngineHandle + EngineListener +
       FrameRingBuffer + AudioObserver, wires them, then awaits the
       OnJoinChannelResult callback via a join-future.
    2. ``push_audio`` calls EngineHandle.push_external_audio synchronously
       (the C++ side releases the GIL during the SDK call); errors raise.
    3. ``drain_inbound_frames`` calls FrameRingBuffer.drain — non-blocking.
    4. ``leave`` calls EngineHandle.leave_channel + .destroy.
    """

    def __init__(self, *, binding: Any) -> None:
        self._binding = binding
        self._engine: Any | None = None
        self._listener: Any | None = None
        self._frame_buffer: Any | None = None
        self._observer: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._join_future: asyncio.Future[tuple[int, str, str, int]] | None = None
        self._on_join_result_cb: JoinResultCallback | None = None
        self._on_error_cb: ErrorCallback | None = None
        # Captured from setup_and_join(); reused per-frame so push_audio
        # never sends ``samplesPerSec=0`` to the SDK (vendor will SIGFPE
        # on integer-divide by zero inside the audio pipeline).
        self._send_sample_rate: int = 0
        self._send_channels: int = 0

    async def setup_and_join(
        self,
        *,
        channel: str,
        token: str,
        uid: str,
        app_id: str,
        send_sample_rate: int,
        send_channels: int,
        gslb_server: str,
        join_timeout: float,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._join_future = self._loop.create_future()
        self._send_sample_rate = int(send_sample_rate)
        self._send_channels = int(send_channels)

        # Build SDK objects.
        b = self._binding
        engine = b.EngineHandle()
        listener = b.EngineListener()
        frame_buffer = b.FrameRingBuffer(64)
        observer = b.AudioObserver(frame_buffer, "")

        listener.set_on_join(self._listener_on_join)
        listener.set_on_leave(self._listener_on_leave)
        listener.set_on_error(self._listener_on_error)

        try:
            engine.create("")
            engine.set_event_listener(listener)
            engine.register_audio_observer(observer)
            engine.enable_audio_frame_observer(True, b.POSITION_PLAYBACK)
            engine.set_external_audio_source(True, send_sample_rate, send_channels)
            engine.publish_local_audio_stream(True)
            engine.publish_local_video_stream(False)
            engine.subscribe_all_remote_audio_streams(True)
            engine.join_channel(
                channel, uid, app_id, token, gslb_server, uid,
            )
        except Exception:
            # Tear down partially-constructed engine before propagating.
            with contextlib.suppress(Exception):
                engine.destroy()
            raise

        # Stash refs so leave() can tear down + observer / listener stay
        # alive for the SDK to call back into.
        self._engine = engine
        self._listener = listener
        self._frame_buffer = frame_buffer
        self._observer = observer

        # Await OnJoinChannelResult.
        try:
            result, ch, user_id, elapsed = await asyncio.wait_for(
                self._join_future, timeout=join_timeout,
            )
        except TimeoutError as exc:
            await self.leave()
            raise RtcError(f"JoinChannel timed out after {join_timeout}s") from exc

        if result != 0:
            await self.leave()
            raise RtcError(
                f"JoinChannel failed: rc={result} (0x{result & 0xFFFFFFFF:08X})"
            )

    async def leave(self) -> None:
        engine = self._engine
        if engine is None:
            return
        # Detach refs first so callbacks fired between leave_channel and
        # destroy are no-ops.
        self._engine = None
        listener = self._listener
        self._listener = None
        self._frame_buffer = None
        self._observer = None

        if listener is not None:
            with contextlib.suppress(Exception):
                listener.clear_callbacks()

        with contextlib.suppress(Exception):
            engine.leave_channel()
        with contextlib.suppress(Exception):
            engine.unregister_audio_observer()
        with contextlib.suppress(Exception):
            engine.destroy()

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        engine = self._engine
        if engine is None:
            raise RtcNotJoined("push_audio before setup_and_join")
        # SDK consumes ``RtcEngineAudioFrame.samplesPerSec`` per frame;
        # passing 0 triggers an integer divide-by-zero inside the audio
        # pipeline (observed: SIGFPE on Linux 3.9.0 SDK during the
        # listen loop of ecs_pcm_loopback_listen.py). Always send the
        # rate captured from ``set_external_audio_source`` at join time.
        if self._send_sample_rate <= 0 or self._send_channels <= 0:
            raise RtcNotJoined("push_audio before setup_and_join (rate/channels unset)")

        # DingRTC SDK push_external_audio expects PCM frames sized to the
        # SDK's internal 10ms grid: ``samples_per_10ms = sample_rate / 100``.
        # Empirically (calls 13-15, mac --dev-no-modem smoke), a single
        # TTS-V3-SSE chunk can be 8-12 KB (~250-400 ms at 16 kHz) — passing
        # that whole blob to push_external_audio raises
        # ``DingRtcError: PushExternalAudioFrame failed`` and aborts the
        # _do_greeting turn. Slice into 10 ms frames here and push one by
        # one; SDK does its own internal queueing across frames.
        frame_size = (self._send_sample_rate // 100) * self._send_channels * 2  # 10 ms × 2 bytes/sample
        if not pcm:
            return 0
        ts = timestamp_ms
        # 10 ms per pushed frame keeps timestamp accounting cheap; callers
        # pass an overall start timestamp and the SDK runs the playback clock.
        ts_step_ms = 10
        frames_pushed = 0
        try:
            for off in range(0, len(pcm), frame_size):
                frame = pcm[off:off + frame_size]
                # Pad the last (potentially short) frame to a full SDK
                # frame so the codec sees a clean 10 ms shape.
                if len(frame) < frame_size:
                    frame = frame + b"\x00" * (frame_size - len(frame))
                engine.push_external_audio(
                    frame,
                    sample_rate=self._send_sample_rate,
                    channels=self._send_channels,
                    bytes_per_sample=2,
                    timestamp=ts,
                )
                ts += ts_step_ms
                frames_pushed += 1
        except self._binding.DingRtcError as exc:
            code = exc.args[0] if exc.args else -1
            logger.warning(
                "dingrtc_push_audio_failed rc=%s msg=%s frame_size=%s "
                "send_rate=%s send_ch=%s total_len=%s frames_pushed=%s",
                code, str(exc), frame_size,
                self._send_sample_rate, self._send_channels,
                len(pcm), frames_pushed,
            )
            # DingRTC SDK uses specific error codes for buffer full;
            # without an authoritative list of "buffer full" rcs in
            # engine_conf.h, treat anything matching a transient pattern
            # as backpressure. Conservative: only the documented
            # ERR_AUDIO_BUFFER_FULL family. Until that constant is
            # confirmed, surface the raw error.
            raise RtcError(str(exc)) from exc
        return 0

    def drain_inbound_frames(self, max_n: int = 8) -> list[Any]:
        buf = self._frame_buffer
        if buf is None:
            return []
        return buf.drain(max_n)

    def set_on_join_result(self, cb: JoinResultCallback | None) -> None:
        self._on_join_result_cb = cb

    def set_on_error(self, cb: ErrorCallback | None) -> None:
        self._on_error_cb = cb

    # ----- SDK-thread callbacks (called via EngineListener) --------------
    #
    # NOTE: These run with the GIL held but on an SDK thread, NOT the
    # asyncio loop thread. We schedule loop work via call_soon_threadsafe.

    def _listener_on_join(
        self, result: int, channel: str, user_id: str, elapsed: int,
    ) -> None:
        loop = self._loop
        fut = self._join_future
        cb = self._on_join_result_cb
        if loop is not None and fut is not None and not fut.done():
            loop.call_soon_threadsafe(fut.set_result, (result, channel, user_id, elapsed))
        if cb is not None and loop is not None:
            loop.call_soon_threadsafe(cb, result, channel, user_id, elapsed)

    def _listener_on_leave(self, result: int) -> None:
        # No-op for now — DingRtcSession.leave handles teardown imperatively.
        # If we ever want explicit OnLeaveChannelResult await, plug it here.
        pass

    def _listener_on_error(self, code: int, message: str) -> None:
        loop = self._loop
        cb = self._on_error_cb
        if cb is not None and loop is not None:
            loop.call_soon_threadsafe(cb, code, message)
