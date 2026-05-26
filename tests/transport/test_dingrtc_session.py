"""DingRtcSession lifecycle and backpressure tests (mock channel).

Spec: engine-rtc-dingrtc-migration § 6.6.

These tests use :class:`InMemoryDingRtcChannel` exclusively — no
pybind11 binding loaded, no SDK shared libs, no platform requirement.
The pybind11-backed :class:`_PybindDingRtcChannel` is verified by the
demo / real smoke scripts (``scripts/dingrtc_*.py``) against the actual
DingRTC service, since C++ binding behavior is not unit-testable
in-process.
"""

from __future__ import annotations

import asyncio

import pytest

from isales_common.audio.rtc import (
    PcmFrame,
    RtcError,
    RtcNotJoined,
    RtcPushBackpressure,
)
from isales_engine.transport.dingrtc import (
    DingRtcSession,
    InMemoryDingRtcChannel,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_session_not_joined_by_default() -> None:
    sess = DingRtcSession(
        channel=InMemoryDingRtcChannel(),
        app_id="o6dpsan9-test",
    )
    assert sess.is_joined is False


async def test_join_then_leave() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join("call-1", "tok-1", "engine-call-1")
    assert sess.is_joined is True
    assert ch.setup_calls[0]["channel"] == "call-1"
    assert ch.setup_calls[0]["app_id"] == "o6dpsan9-test"
    assert ch.setup_calls[0]["gslb_server"] == "https://gslb.dingrtc.com"
    await sess.leave()
    assert sess.is_joined is False


async def test_join_twice_raises() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join("call-1", "tok-1", "engine-call-1")
    with pytest.raises(RtcError, match="already joined"):
        await sess.join("call-2", "tok-2", "engine-call-2")
    await sess.leave()


async def test_leave_is_idempotent() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join("call-1", "tok-1", "engine-call-1")
    await sess.leave()
    # Second leave is a no-op.
    await sess.leave()


async def test_join_passes_send_sample_rate_to_channel() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join(
        "call-1", "tok-1", "engine-call-1",
        send_sample_rate=8000, send_channels=1,
    )
    assert ch.setup_calls[0]["send_sample_rate"] == 8000
    await sess.leave()


async def test_join_failure_surfaces_rtc_error() -> None:
    # auto_complete_join=False lets us force the failure path.
    ch = InMemoryDingRtcChannel(auto_complete_join=False)
    sess = DingRtcSession(
        channel=ch, app_id="o6dpsan9-test", join_timeout=0.5,
    )

    async def fail_join() -> None:
        # Give setup_and_join a tick to register its future.
        await asyncio.sleep(0.01)
        ch.complete_join(result=0x02010205)  # ERR_JOIN_BAD_TOKEN

    fail_task = asyncio.create_task(fail_join())
    with pytest.raises(RtcError, match="0x02010205"):
        await sess.join("call-1", "tok-1", "engine-call-1")
    await fail_task
    assert sess.is_joined is False


async def test_join_timeout_surfaces_rtc_error() -> None:
    ch = InMemoryDingRtcChannel(auto_complete_join=False)
    sess = DingRtcSession(
        channel=ch, app_id="o6dpsan9-test", join_timeout=0.05,
    )
    with pytest.raises(asyncio.TimeoutError):
        await sess.join("call-1", "tok-1", "engine-call-1")


# ---------------------------------------------------------------------------
# Inbound PCM
# ---------------------------------------------------------------------------


async def test_audio_frames_before_join_raises() -> None:
    sess = DingRtcSession(
        channel=InMemoryDingRtcChannel(), app_id="o6dpsan9-test",
    )
    with pytest.raises(RtcNotJoined):
        async for _ in sess.audio_frames():
            break


async def test_inbound_frame_flows_to_audio_frames() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(
        channel=ch, app_id="o6dpsan9-test",
        drain_interval_ms=2.0,  # speed up the drain loop
    )
    await sess.join("call-1", "tok-1", "engine-call-1")
    ch.inject_remote_pcm("edge-call-1", b"\x01\x02\x03\x04", sample_rate=16000)

    async def consume_one() -> PcmFrame:
        async for frame in sess.audio_frames():
            return frame
        raise AssertionError("audio_frames terminated before yielding a frame")

    frame = await asyncio.wait_for(consume_one(), timeout=1.0)
    assert frame.sender_uid == "edge-call-1"
    assert frame.pcm == b"\x01\x02\x03\x04"
    assert frame.sample_rate == 16000

    await sess.leave()


async def test_audio_frames_terminates_on_leave() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(
        channel=ch, app_id="o6dpsan9-test", drain_interval_ms=2.0,
    )
    await sess.join("call-1", "tok-1", "engine-call-1")

    async def consume_all() -> int:
        n = 0
        async for _ in sess.audio_frames():
            n += 1
        return n

    consumer = asyncio.create_task(consume_all())
    await asyncio.sleep(0.05)  # let drainer warm up
    await sess.leave()
    n = await asyncio.wait_for(consumer, timeout=1.0)
    assert n == 0  # no frames were injected


# ---------------------------------------------------------------------------
# Outbound PCM
# ---------------------------------------------------------------------------


async def test_push_audio_before_join_raises() -> None:
    sess = DingRtcSession(
        channel=InMemoryDingRtcChannel(), app_id="o6dpsan9-test",
    )
    with pytest.raises(RtcNotJoined):
        await sess.push_audio(b"\x00" * 32, timestamp_ms=0)


async def test_push_audio_records_frames() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join("call-1", "tok-1", "engine-call-1")
    await sess.push_audio(b"\x00\x01" * 16, timestamp_ms=100)
    await sess.push_audio(b"\x02\x03" * 16, timestamp_ms=120)
    assert len(ch.pushed) == 2
    assert ch.pushed[0] == (b"\x00\x01" * 16, 100)
    assert ch.pushed[1] == (b"\x02\x03" * 16, 120)
    await sess.leave()


async def test_push_audio_surfaces_backpressure() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join("call-1", "tok-1", "engine-call-1")
    ch.reject_next_push = True
    with pytest.raises(RtcPushBackpressure):
        await sess.push_audio(b"\x00" * 32, timestamp_ms=0)
    # After the rejected push, subsequent pushes are accepted again.
    await sess.push_audio(b"\xff" * 32, timestamp_ms=20)
    assert ch.pushed == [(b"\xff" * 32, 20)]
    await sess.leave()


async def test_push_audio_after_leave_raises() -> None:
    ch = InMemoryDingRtcChannel()
    sess = DingRtcSession(channel=ch, app_id="o6dpsan9-test")
    await sess.join("call-1", "tok-1", "engine-call-1")
    await sess.leave()
    with pytest.raises(RtcNotJoined):
        await sess.push_audio(b"\x00" * 32, timestamp_ms=0)
