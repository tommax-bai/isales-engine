"""In-memory :class:`DingRtcChannelProtocol` implementation for tests.

Mirrors the surface of :class:`_PybindDingRtcChannel` without loading the
C++ binding. Replaces the legacy ``InMemorySdkChannel`` that lived in
``isales_engine.transport.aliyun_rtc``.

Test helpers (`inject_remote_pcm`, `complete_join`, `force_join_error`)
let test code drive the channel through the lifecycle states without
spawning real SDK threads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from isales_common.audio.rtc import (
    RtcError,
    RtcNotJoined,
    RtcPushBackpressure,
)

__all__ = ["InMemoryDingRtcChannel", "InMemoryPcmFrame"]


class InMemoryPcmFrame:
    """Duck-typed equivalent of the pybind11 ``PcmFrame`` C++ struct.

    The session's drainer reads ``pcm`` / ``sample_rate`` / ``channels`` /
    ``remote_uid`` / ``timestamp`` attrs — any object with those fields
    works.
    """

    def __init__(
        self,
        *,
        pcm: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        bytes_per_sample: int = 2,
        num_samples: int = 0,
        remote_uid: str = "",
        timestamp: int = 0,
    ) -> None:
        self.pcm = pcm
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = bytes_per_sample
        self.num_samples = num_samples or (len(pcm) // (bytes_per_sample * channels))
        self.remote_uid = remote_uid
        self.timestamp = timestamp


class InMemoryDingRtcChannel:
    """Test double for :class:`DingRtcChannelProtocol`.

    Behavior:

    - :meth:`setup_and_join` does not auto-resolve. Tests call
      :meth:`complete_join(result=0)` after asserting setup state to
      simulate the SDK's OnJoinChannelResult callback.
    - :meth:`push_audio` records frames into :attr:`pushed`. Set
      :attr:`reject_next_push` to ``True`` to force the next push to
      raise :class:`RtcPushBackpressure`.
    - :meth:`inject_remote_pcm` appends to an internal queue that
      :meth:`drain_inbound_frames` will return.
    """

    def __init__(
        self,
        *,
        auto_complete_join: bool = True,
        join_delay_s: float = 0.0,
    ) -> None:
        self._auto_complete_join = auto_complete_join
        self._join_delay_s = join_delay_s
        self._joined = False
        self._inbound_frames: list[InMemoryPcmFrame] = []
        self._on_join_result_cb: Any | None = None
        self._on_error_cb: Any | None = None
        self._join_fut: asyncio.Future[tuple[int, str, str, int]] | None = None
        self._join_channel: str = ""
        self._join_uid: str = ""

        # Tests inspect these:
        self.pushed: list[tuple[bytes, int]] = []
        self.reject_next_push = False
        self.setup_calls: list[dict[str, Any]] = []

    # ----- DingRtcChannelProtocol -----------------------------------------

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
        self.setup_calls.append(
            {
                "channel": channel, "token": token, "uid": uid,
                "app_id": app_id, "send_sample_rate": send_sample_rate,
                "send_channels": send_channels, "gslb_server": gslb_server,
                "join_timeout": join_timeout,
            }
        )
        self._join_channel = channel
        self._join_uid = uid
        loop = asyncio.get_running_loop()
        self._join_fut = loop.create_future()

        if self._auto_complete_join:
            if self._join_delay_s > 0:
                await asyncio.sleep(self._join_delay_s)
            self.complete_join(result=0)

        result, _ch, _uid, _elapsed = await asyncio.wait_for(
            self._join_fut, timeout=join_timeout,
        )
        if result != 0:
            raise RtcError(
                f"JoinChannel failed: rc={result} (0x{result & 0xFFFFFFFF:08X})"
            )
        self._joined = True

    async def leave(self) -> None:
        self._joined = False

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        if not self._joined:
            raise RtcNotJoined("push_audio before setup_and_join")
        if self.reject_next_push:
            self.reject_next_push = False
            raise RtcPushBackpressure("simulated buffer full")
        self.pushed.append((pcm, timestamp_ms))
        return 0

    def drain_inbound_frames(self, max_n: int = 8) -> list[InMemoryPcmFrame]:
        if not self._inbound_frames:
            return []
        take = self._inbound_frames[:max_n]
        self._inbound_frames = self._inbound_frames[max_n:]
        return take

    def set_on_join_result(self, cb: Any | None) -> None:
        self._on_join_result_cb = cb

    def set_on_error(self, cb: Any | None) -> None:
        self._on_error_cb = cb

    # ----- test helpers ----------------------------------------------------

    def complete_join(
        self, *, result: int = 0, elapsed_ms: int = 42,
    ) -> None:
        """Simulate the SDK OnJoinChannelResult callback."""
        if self._join_fut is not None and not self._join_fut.done():
            self._join_fut.set_result(
                (result, self._join_channel, self._join_uid, elapsed_ms)
            )

    def force_join_error(self, code: int, message: str = "") -> None:
        """Simulate the SDK reporting an error during join."""
        if self._on_error_cb is not None:
            self._on_error_cb(code, message)
        # If join hasn't completed yet, fail the future with the code:
        if self._join_fut is not None and not self._join_fut.done():
            self._join_fut.set_result(
                (code, self._join_channel, self._join_uid, 0)
            )

    def inject_remote_pcm(
        self,
        uid: str,
        pcm: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        timestamp: int = 0,
    ) -> None:
        """Simulate the SDK delivering one inbound PCM frame."""
        self._inbound_frames.append(
            InMemoryPcmFrame(
                pcm=pcm,
                sample_rate=sample_rate,
                channels=channels,
                remote_uid=uid,
                timestamp=timestamp,
            )
        )
