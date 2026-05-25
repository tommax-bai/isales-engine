"""End-to-end tests for `_AliyunArtcChannel` with a stub vendor SDK.

Spec: device-hardware § Requirement: 云端 isales-engine
``transport/aliyun_rtc.py`` 通过专用 SDK 驱动线程承载 vendor recvCoroutine.

The vendor SDK binary is Linux-only and not on the dev machine. These
tests substitute a small in-process stub that mirrors the vendor's
threading contract (``CreateAliRTCEngine`` does init on whatever
``asyncio.get_event_loop()`` returns, ``JoinChannel`` does
``loop.run_until_complete(...)``, ``OnJoinChannelResult`` fires on the
same thread after ``JoinChannel`` returns).

The point is to assert the wrapper:

1. Spawns a dedicated driver thread for vendor calls.
2. Pumps the driver loop so vendor coroutines that the wrapper-as-init
   registers can progress.
3. Tears the driver thread down on ``leave`` and on ``join`` failure.
4. Surfaces ``DriverQueueFull`` as ``RtcPushBackpressure``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from isales_common.audio.rtc import RtcPushBackpressure

from isales_engine.transport import _rtc_sdk
from isales_engine.transport._rtc_sdk import _AliyunArtcChannel


# --------------------------------------------------------------------------
# Stub vendor modules
# --------------------------------------------------------------------------


class _Enum:
    """Tiny enum-stand-in that mimics the vendor's `.value` attribute."""

    def __init__(self, value: Any) -> None:
        self.value = value


class _StubEnums:
    """Minimal stand-in for AliRTCLinuxSdkDefine constants the wrapper reads."""

    class JoinChannelConfig:
        def __init__(self) -> None:
            self.channelProfile = None
            self.subscribeAudioFormat = None
            self.subscribeVideoFormat = None
            self.isAudioOnly = False
            self.publishAvsyncMode = None
            self.publishAvsyncWithPtsMaxAudioCacheSize = 0
            self.publishAvsyncWithPtsMaxVideoCacheSize = 0
            self.subscribeMode = None
            self.publishMode = None
            self.enableTtsCallback = False

    class ChannelProfile:
        ChannelProfileInteractiveLive = _Enum(1)

    class AudioFormat:
        AudioFormatPcmBeforMixing = _Enum(2)

    class VideoFormat:
        VideoFormatH264 = _Enum(3)

    class PublishAvsyncMode:
        PublishAvsyncWithPts = _Enum(4)

    class SubscribeMode:
        SubscribeAutomatically = _Enum(5)

    class PublishMode:
        PublishAutomatically = _Enum(6)

    class AliEngineClientRole:
        AliEngineClientRoleInteractive = _Enum(7)


class _StubEngine:
    """A minimal vendor engine that records calls and fires
    OnJoinChannelResult exactly once when JoinChannel is invoked.

    Records the thread name and event-loop identity for every method so
    tests can assert "everything ran on the driver thread, on the same
    loop the wrapper set with set_event_loop()".
    """

    def __init__(
        self,
        handler: Any,
        *,
        join_result_code: int = 0,
        fail_on_join: bool = False,
    ) -> None:
        self._handler = handler
        self._join_result_code = join_result_code
        self._fail_on_join = fail_on_join
        self.calls: list[str] = []
        self.thread_names: list[str] = []
        self.audio_pushes: list[tuple[bytes, int, int]] = []
        self.released = False

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.thread_names.append(threading.current_thread().name)

    def PublishLocalAudioStream(self, on: bool) -> None:  # noqa: N802 (vendor name)
        self._record(f"PublishLocalAudioStream({on})")

    def SetExternalAudioSource(  # noqa: N802
        self,
        on: bool,
        *,
        sampleRate: int,  # noqa: N803 (vendor name)
        channelsPerFrame: int,  # noqa: N803
    ) -> None:
        self._record(f"SetExternalAudioSource({on},{sampleRate},{channelsPerFrame})")

    def SetClientRole(self, role: Any) -> None:  # noqa: N802
        self._record("SetClientRole")

    def JoinChannel(  # noqa: N802
        self,
        token: str,
        channel: str,
        uid: str,
        uname: str,
        config: Any,
    ) -> int:
        self._record(f"JoinChannel({channel},{uid})")
        if self._fail_on_join:
            return -1
        # Vendor's JoinChannel internally does loop.run_until_complete —
        # simulate that constraint by requiring a running asyncio loop to
        # exist on this thread. If the wrapper accidentally calls us on
        # a thread that has no event loop, this will raise.
        loop = asyncio.get_event_loop()
        assert loop is not None
        # Fire the callback now (vendor fires it async after a roundtrip;
        # for the stub, doing it inline is sufficient to exercise the
        # call_soon_threadsafe marshaling path on the wrapper side).
        self._handler.OnJoinChannelResult(self._join_result_code, channel, uid)
        return 0

    def LeaveChannel(self) -> None:  # noqa: N802
        self._record("LeaveChannel")

    def Release(self) -> None:  # noqa: N802
        self._record("Release")
        self.released = True

    def PushExternalAudioFrameRawData(  # noqa: N802
        self,
        pcm: bytes,
        length: int,
        timestamp_ms: int,
    ) -> int:
        self._record("PushExternalAudioFrameRawData")
        self.audio_pushes.append((pcm, length, timestamp_ms))
        return 0


class _StubArtcModule:
    """Replaces the vendor's AliRTCEngine module."""

    class EngineEventHandlerInterface:
        """Mirrors the vendor base class shape; we expect subclasses to
        override the OnFoo methods the wrapper uses."""

        def OnJoinChannelResult(self, result: int, channel: str, userId: str) -> None:  # noqa: N802, N803
            pass

        def OnLeaveChannelResult(self, result: int) -> None:  # noqa: N802
            pass

        def OnSubscribeAudioFrame(self, uid: str, frame: Any) -> None:  # noqa: N802
            pass

        def OnPushAudioFrameBufferFull(self, isFull: bool) -> None:  # noqa: N802, N803
            pass

        def OnError(self, error_code: Any) -> None:  # noqa: N802
            pass

        def OnWarning(self, warning_code: Any) -> None:  # noqa: N802
            pass

    def __init__(self) -> None:
        self.last_engine: _StubEngine | None = None
        self.fail_on_join = False
        self.join_result_code = 0

    def CreateAliRTCEngine(  # noqa: N802
        self,
        handler: Any,
        port_min: int,
        port_max: int,
        work_dir: str,
        core_service_path: str,
        h5mode: bool,
        extra: str,
    ) -> _StubEngine:
        # Vendor's CreateAliRTCEngine internally does
        # loop.run_until_complete(InitializeEngine) — assert a loop is
        # current to mirror that requirement.
        loop = asyncio.get_event_loop()
        assert loop is not None
        eng = _StubEngine(
            handler,
            join_result_code=self.join_result_code,
            fail_on_join=self.fail_on_join,
        )
        self.last_engine = eng
        return eng


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_channel(
    *,
    fail_on_join: bool = False,
    join_result_code: int = 0,
) -> tuple[_AliyunArtcChannel, _StubArtcModule]:
    stub_artc = _StubArtcModule()
    stub_artc.fail_on_join = fail_on_join
    stub_artc.join_result_code = join_result_code
    ch = _AliyunArtcChannel(
        app_id="stub-app",
        engine_module=stub_artc,
        define_module=_StubEnums,
    )
    return ch, stub_artc


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_spawns_driver_thread_and_runs_vendor_calls_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Speed up pumps so the test isn't waiting 100ms per iteration.
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "5")
    ch, artc = _make_channel()
    try:
        await ch.join(
            "channel-x", "tok", "engine-x",
            send_sample_rate=16000, send_channels=1,
        )
        assert artc.last_engine is not None
        # All vendor calls MUST have run on the driver thread (named
        # ``artc-driver-<uid>``), not on the caller's main thread.
        assert artc.last_engine.thread_names
        assert all(
            name == "artc-driver-engine-x"
            for name in artc.last_engine.thread_names
        )
    finally:
        await ch.leave()


@pytest.mark.asyncio
async def test_join_failure_tears_driver_down_no_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "5")
    ch, _ = _make_channel(fail_on_join=True)
    with pytest.raises(RuntimeError, match="JoinChannel returned non-zero"):
        await ch.join(
            "channel-x", "tok", "engine-x",
            send_sample_rate=16000, send_channels=1,
        )
    # join() must clean up its own driver on failure — see design
    # Decision 5; otherwise mac-dev / ECS-long-run leaks threads per
    # failed call.
    assert ch._driver is None  # noqa: SLF001 — verifying lifecycle


@pytest.mark.asyncio
async def test_join_nonzero_result_code_raises_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "5")
    ch, _ = _make_channel(join_result_code=1037)
    with pytest.raises(RuntimeError, match="result=1037"):
        await ch.join(
            "channel-x", "tok", "engine-x",
            send_sample_rate=16000, send_channels=1,
        )
    assert ch._driver is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_leave_calls_vendor_release_then_stops_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "5")
    ch, artc = _make_channel()
    await ch.join(
        "channel-x", "tok", "engine-x",
        send_sample_rate=16000, send_channels=1,
    )
    eng = artc.last_engine
    assert eng is not None
    await ch.leave()
    assert eng.released is True
    assert "LeaveChannel" in eng.calls
    assert "Release" in eng.calls
    assert ch._driver is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_leave_idempotent_before_join() -> None:
    # Channel constructed but never joined — leave must be a clean no-op,
    # NOT raise and NOT spawn a driver thread.
    ch, _ = _make_channel()
    await ch.leave()
    assert ch._driver is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_push_audio_dispatches_to_driver_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "5")
    ch, artc = _make_channel()
    try:
        await ch.join(
            "channel-x", "tok", "engine-x",
            send_sample_rate=16000, send_channels=1,
        )
        eng = artc.last_engine
        assert eng is not None
        assert await ch.push_audio(b"\x01\x02", timestamp_ms=42) == 0
        # Recorded on the driver thread.
        assert eng.audio_pushes == [(b"\x01\x02", 2, 42)]
        # And the push ran on the driver thread, not the caller.
        assert eng.thread_names[-1] == "artc-driver-engine-x"
    finally:
        await ch.leave()


@pytest.mark.asyncio
async def test_push_audio_queue_full_raises_rtc_push_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the driver-thread command queue is at its cap, push_audio
    MUST raise :class:`RtcPushBackpressure` rather than block the loop."""
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "50")
    ch, artc = _make_channel()
    await ch.join(
        "channel-x", "tok", "engine-x",
        send_sample_rate=16000, send_channels=1,
    )
    eng = artc.last_engine
    assert eng is not None

    # Shrink the cap to 1 so we can saturate easily, and block the
    # driver thread on a long-running stub push so the queue fills.
    # ``in_slow_push`` lets us positively observe "the driver thread
    # has begun executing the blocker", which is more robust than
    # polling queue depth (which can be 0 both before the cmd was
    # submitted and after it was drained).
    block = threading.Event()
    in_slow_push = threading.Event()
    original_push = eng.PushExternalAudioFrameRawData

    def slow_push(pcm: bytes, length: int, timestamp_ms: int) -> int:
        in_slow_push.set()
        block.wait(timeout=2.0)
        return original_push(pcm, length, timestamp_ms)

    eng.PushExternalAudioFrameRawData = slow_push  # type: ignore[method-assign]

    # Shrink the driver queue cap so saturation is easy to hit. The
    # driver's queue is fixed at construction; the easiest way to test
    # saturation against this wrapper is to reach into the channel's
    # driver and replace its queue. That's a knowingly internal hook —
    # acceptable for the spec scenario this test covers.
    import queue as _queue

    assert ch._driver is not None  # noqa: SLF001
    ch._driver._cmd_q = _queue.Queue(maxsize=1)  # noqa: SLF001

    try:
        # First push: driver thread picks it up and blocks inside
        # slow_push. Wait until slow_push has actually started — only
        # then is the queue guaranteed empty AND the driver guaranteed
        # busy, which is the precondition for the cap-saturation race
        # this test asserts.
        first = asyncio.create_task(ch.push_audio(b"\xaa", timestamp_ms=0))
        deadline = time.monotonic() + 2.0
        while not in_slow_push.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert in_slow_push.is_set(), "driver did not enter slow_push"

        # Second push: queue empty, driver busy → submit succeeds and
        # sits waiting for the driver to free up.
        second = asyncio.create_task(ch.push_audio(b"\xbb", timestamp_ms=1))
        deadline = time.monotonic() + 1.0
        while ch._driver.queue_size() != 1 and time.monotonic() < deadline:  # noqa: SLF001
            await asyncio.sleep(0.005)
        assert ch._driver.queue_size() == 1  # noqa: SLF001

        # Third push: queue is now at cap (1) → MUST raise.
        with pytest.raises(RtcPushBackpressure):
            await ch.push_audio(b"\xcc", timestamp_ms=2)

        # Unblock and let the first two complete normally.
        block.set()
        assert await asyncio.wait_for(first, timeout=1.0) == 0
        assert await asyncio.wait_for(second, timeout=1.0) == 0
    finally:
        block.set()
        await ch.leave()


@pytest.mark.asyncio
async def test_callbacks_marshal_back_to_main_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OnSubscribeAudioFrame fires on the driver thread; the inbound
    callback the wrapper registered MUST be invoked on that same thread
    (engine-session marshaling to the main loop is the next layer up,
    handled in AliyunRtcSession). This test pins the contract.
    """
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "5")
    ch, artc = _make_channel()

    recorded: list[tuple[str, str]] = []  # (uid, thread_name)

    def on_inbound(uid: str, pcm: bytes, sr: int, ts: int) -> None:
        recorded.append((uid, threading.current_thread().name))

    ch.on_inbound_frame(on_inbound)

    await ch.join(
        "channel-x", "tok", "engine-x",
        send_sample_rate=16000, send_channels=1,
    )
    try:
        eng = artc.last_engine
        assert eng is not None
        handler = eng._handler  # noqa: SLF001 — accessing stub's stored handler

        # Synthesize an OnSubscribeAudioFrame from the driver thread by
        # submitting it as a vendor "call" — that runs the handler on
        # the same thread the real vendor would.
        class _StubPcm:
            pcmBuf_ = b"\x10\x20"
            sample_rates_ = 16000

        class _StubFrame:
            pcm = _StubPcm()

        assert ch._driver is not None  # noqa: SLF001
        await ch._driver.call(  # noqa: SLF001
            handler.OnSubscribeAudioFrame, "edge-x", _StubFrame(),
        )
        assert len(recorded) == 1
        uid, tname = recorded[0]
        assert uid == "edge-x"
        assert tname == "artc-driver-engine-x"
    finally:
        await ch.leave()


@pytest.mark.asyncio
async def test_vendor_modules_attribute_path_unchanged() -> None:
    """Regression: the module exports stayed backwards-compatible — the
    arch-cloud-edge-split spec relies on importing _AliyunArtcChannel
    + vendor_channel_factory from this exact path."""
    assert hasattr(_rtc_sdk, "_AliyunArtcChannel")
    assert hasattr(_rtc_sdk, "vendor_channel_factory")
    assert hasattr(_rtc_sdk, "SdkChannel")
    assert hasattr(_rtc_sdk, "SdkLoadError")
