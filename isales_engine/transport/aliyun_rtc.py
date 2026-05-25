"""Cloud-side :class:`RtcSession` backed by the Aliyun ARTC SDK.

Spec: device-hardware § Requirement: 云端 engine 的 ARTC SDK 接入;
      service-communication § Requirement: 云-边媒体面 (阿里 RTC PaaS).

This is the cloud half of the cloud↔edge media plane. One
:class:`AliyunRtcSession` instance per active call:

- Joins the call's RTC channel as ``"engine-{call_id}"``.
- Surfaces inbound PCM from the edge audio-bridge via the
  :meth:`audio_frames` async iterator.
- Pushes outbound TTS PCM via :meth:`push_audio` with native asyncio
  backpressure (awaits the SDK's drain signal rather than raising).

The session is **not** thread-safe: the SDK invokes callbacks from its
own C thread; we marshal those onto the engine event loop via
``loop.call_soon_threadsafe`` so business code stays single-threaded
asyncio.

Vendor SDK binding lives in :mod:`isales_engine.transport._rtc_sdk`. This
module never imports the vendor module directly — concrete
:class:`SdkChannel` is either constructed via
:func:`_rtc_sdk.vendor_channel_factory` (production) or injected by tests
(:class:`InMemorySdkChannel` below).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Self

from isales_common.audio.rtc import (
    PcmFrame,
    RtcError,
    RtcNotJoined,
    RtcSession,
)

from isales_engine.transport._rtc_sdk import (
    BufferStateCallback,
    InboundFrameCallback,
    SdkChannel,
    SdkLoadError,
    vendor_channel_factory,
)

__all__ = ["AliyunRtcSession", "InMemorySdkChannel", "SdkLoadError"]


class AliyunRtcSession(RtcSession):
    """RtcSession backed by a vendor :class:`SdkChannel`.

    Args:
        channel: vendor-agnostic channel adaptor. In tests pass an
            :class:`InMemorySdkChannel`; in production construct via
            :func:`_rtc_sdk.vendor_channel_factory`.
        inbound_buffer_size: bound on the inbound PCM queue. Backpressure
            for ASR consumers — they are expected to keep up; a slow
            consumer will block the SDK callback thread once the queue
            fills, which is then visible as audio loss on the wire.
    """

    def __init__(
        self,
        channel: SdkChannel,
        *,
        inbound_buffer_size: int = 1024,
    ) -> None:
        self._channel = channel
        self._inbound: asyncio.Queue[PcmFrame | None] = asyncio.Queue(
            maxsize=inbound_buffer_size,
        )
        # SDK starts with no backpressure; flip to .clear() when the SDK
        # signals buffer-full, .set() when it drains.
        self._drain_event = asyncio.Event()
        self._drain_event.set()
        self._joined = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_sample_rate = 16000

    # ----- factory --------------------------------------------------------

    @classmethod
    def production(cls, *, app_id: str) -> Self:
        """Build a session wired to the real vendor SDK.

        Raises :class:`SdkLoadError` if the vendor SDK module is not
        importable on this host. Tests SHOULD construct ``cls(channel=
        InMemorySdkChannel())`` directly.
        """
        return cls(channel=vendor_channel_factory(app_id))

    # ----- lifecycle -------------------------------------------------------

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

        # Cache the loop so SDK-thread callbacks can marshal onto it.
        self._loop = asyncio.get_running_loop()
        self._send_sample_rate = send_sample_rate

        # Register callbacks BEFORE join — the SDK may start delivering
        # frames synchronously inside the join call once the channel
        # comes up.
        self._channel.on_inbound_frame(self._on_inbound_frame)
        self._channel.on_buffer_state(self._on_buffer_state)

        # Vendor SDK's CreateAliRTCEngine (called inside _channel.join)
        # invokes asyncio.get_event_loop() to cache a loop reference for
        # its async callbacks. That call raises in Python 3.10+ when run
        # from a worker thread without a current loop, so we MUST call it
        # from the asyncio main thread (vendor's demo.py does the same).
        # CreateAliRTCEngine is IPC setup to the AliRtcCoreService sidecar
        # process, not heavy compute — keep it on the loop.
        self._channel.join(
            channel,
            token,
            uid,
            send_sample_rate=send_sample_rate,
            send_channels=send_channels,
        )
        self._joined = True

    async def leave(self) -> None:
        if not self._joined:
            return
        self._joined = False
        # Tell the SDK to release the channel; off-loop because the vendor
        # call can block.
        await asyncio.to_thread(self._channel.leave)
        # Sentinel ends any in-flight audio_frames() iterator.
        await self._inbound.put(None)
        # Release anyone waiting on outbound drain.
        self._drain_event.set()

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

    def _on_inbound_frame(
        self,
        uid: str,
        pcm: bytes,
        sample_rate: int,
        timestamp_ms: int,
    ) -> None:
        """Vendor SDK callback. Runs on the SDK's C thread."""
        if self._loop is None or not self._joined:
            return
        frame = PcmFrame(
            sender_uid=uid,
            pcm=pcm,
            sample_rate=sample_rate,
            channels=1,
            timestamp_ms=timestamp_ms,
        )
        # put_nowait + threadsafe scheduling: if the asyncio queue is full,
        # asyncio.Queue.put_nowait raises QueueFull. The cross-thread call
        # logs and drops rather than blocking the SDK thread (which would
        # stall all RTC delivery, not just ours).
        self._loop.call_soon_threadsafe(self._try_put_inbound, frame)

    def _try_put_inbound(self, frame: PcmFrame) -> None:
        """Run on the event loop thread; drop on QueueFull instead of
        blocking the SDK callback indefinitely."""
        with contextlib.suppress(asyncio.QueueFull):
            self._inbound.put_nowait(frame)

    # ----- outbound -------------------------------------------------------

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> None:
        if not self._joined:
            raise RtcNotJoined("push_audio() called before join()")

        # Natural backpressure: if the SDK reported its outbound buffer
        # full, await drain before pushing more. This is the "push_audio
        # absorbs backpressure" contract from
        # isales_common.audio.rtc.RtcSession.push_audio.
        if not self._drain_event.is_set():
            await self._drain_event.wait()

        # Try once. If the SDK accepts but reports full (a transient
        # signal — the frame still went in but no more space), clear the
        # drain event so the NEXT push awaits.
        ret = self._channel.push_audio(pcm, timestamp_ms=timestamp_ms)
        if ret != 0:
            # Vendor's ERR_AUDIO_BUFFER_FULL convention: the frame was
            # rejected; signal backpressure for the next call.
            self._drain_event.clear()

    def _on_buffer_state(self, is_full: bool) -> None:
        """Vendor SDK callback for outbound buffer transitions. Runs on
        the SDK's C thread."""
        if self._loop is None:
            return
        if is_full:
            self._loop.call_soon_threadsafe(self._drain_event.clear)
        else:
            self._loop.call_soon_threadsafe(self._drain_event.set)


# =============================================================================
# In-memory SDK channel for tests
# =============================================================================


class InMemorySdkChannel(SdkChannel):
    """Test double for :class:`SdkChannel`.

    Provides direct hooks for tests to:

    - Simulate inbound frames (:meth:`inject_remote_pcm`).
    - Simulate the SDK's buffer-full transitions
      (:meth:`signal_buffer_full` / :meth:`signal_buffer_drained`).
    - Observe pushed PCM (:attr:`pushed`).
    - Force :meth:`push_audio` to report buffer-full on its return value
      (:attr:`reject_next_push`), independent of the callback.

    No thread is involved — callbacks run inline on the caller's thread.
    The engine wrapper still marshals via ``call_soon_threadsafe`` (a no-op
    on the same thread, but the contract still holds).
    """

    def __init__(self) -> None:
        self._inbound_cb: InboundFrameCallback | None = None
        self._buffer_cb: BufferStateCallback | None = None
        self._joined = False
        self.pushed: list[tuple[bytes, int]] = []
        self.reject_next_push = False

    # ----- SdkChannel surface --------------------------------------------

    def join(
        self,
        channel: str,
        token: str,
        uid: str,
        *,
        send_sample_rate: int,
        send_channels: int,
    ) -> None:
        self._joined = True

    def leave(self) -> None:
        self._joined = False

    def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        if not self._joined:
            raise RuntimeError("push_audio() before join() (test setup bug)")
        if self.reject_next_push:
            self.reject_next_push = False
            return 1  # vendor ERR_AUDIO_BUFFER_FULL
        self.pushed.append((pcm, timestamp_ms))
        return 0

    def on_inbound_frame(self, callback: InboundFrameCallback) -> None:
        self._inbound_cb = callback

    def on_buffer_state(self, callback: BufferStateCallback) -> None:
        self._buffer_cb = callback

    # ----- test helpers --------------------------------------------------

    def inject_remote_pcm(
        self,
        uid: str,
        pcm: bytes,
        *,
        sample_rate: int = 16000,
        timestamp_ms: int = 0,
    ) -> None:
        """Simulate an SDK ``OnSubscribeAudioFrame`` callback."""
        if self._inbound_cb is None:
            raise RuntimeError("inject_remote_pcm before on_inbound_frame")
        self._inbound_cb(uid, pcm, sample_rate, timestamp_ms)

    def signal_buffer_full(self) -> None:
        if self._buffer_cb is None:
            raise RuntimeError("signal_buffer_full before on_buffer_state")
        self._buffer_cb(True)

    def signal_buffer_drained(self) -> None:
        if self._buffer_cb is None:
            raise RuntimeError("signal_buffer_drained before on_buffer_state")
        self._buffer_cb(False)
