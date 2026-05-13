"""AliyunRtcSession lifecycle and backpressure tests.

Spec: device-hardware § Requirement: 云端 engine 的 ARTC SDK 接入.

The vendor SDK is not available in CI / dev; tests use
:class:`InMemorySdkChannel` (from the same module) which lets us drive
inbound frames and buffer-state transitions directly.
"""

from __future__ import annotations

import asyncio

import pytest
from isales_common.audio.rtc import PcmFrame, RtcError, RtcNotJoined, RtcSession

from isales_engine.transport.aliyun_rtc import (
    AliyunRtcSession,
    InMemorySdkChannel,
)


def _silence(ms: int, *, sample_rate: int = 16000) -> bytes:
    return b"\x00\x00" * (sample_rate * ms // 1000)


# --------------------------------------------------------------------------
# ABC conformance
# --------------------------------------------------------------------------


def test_aliyun_session_satisfies_rtc_session_abc() -> None:
    sess = AliyunRtcSession(channel=InMemorySdkChannel())
    assert isinstance(sess, RtcSession)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_marks_joined_and_drives_sdk() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    assert sess.is_joined is False

    await sess.join("c-1", "tok", "engine-c-1")
    assert sess.is_joined is True
    assert ch._joined is True  # noqa: SLF001 — internal attribute is part of the test double


@pytest.mark.asyncio
async def test_double_join_raises() -> None:
    sess = AliyunRtcSession(channel=InMemorySdkChannel())
    await sess.join("c-1", "tok", "engine-c-1")
    with pytest.raises(RtcError):
        await sess.join("c-1", "tok", "engine-c-1")


@pytest.mark.asyncio
async def test_leave_is_idempotent_and_releases_sdk() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")
    await sess.leave()
    assert sess.is_joined is False
    assert ch._joined is False  # noqa: SLF001
    # Idempotent.
    await sess.leave()


@pytest.mark.asyncio
async def test_push_before_join_raises() -> None:
    sess = AliyunRtcSession(channel=InMemorySdkChannel())
    with pytest.raises(RtcNotJoined):
        await sess.push_audio(_silence(20), timestamp_ms=0)


@pytest.mark.asyncio
async def test_audio_frames_before_join_raises() -> None:
    sess = AliyunRtcSession(channel=InMemorySdkChannel())
    with pytest.raises(RtcNotJoined):
        async for _ in sess.audio_frames():
            pass


# --------------------------------------------------------------------------
# Inbound flow
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_inbound_callback_delivers_pcm_frame() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")

    payload = _silence(20)
    ch.inject_remote_pcm("edge-c-1", payload, timestamp_ms=42)

    iterator = sess.audio_frames()
    frame = await asyncio.wait_for(anext(iterator), timeout=1.0)
    assert isinstance(frame, PcmFrame)
    assert frame.sender_uid == "edge-c-1"
    assert frame.pcm == payload
    assert frame.timestamp_ms == 42
    assert frame.sample_rate == 16000

    await sess.leave()


@pytest.mark.asyncio
async def test_leave_terminates_audio_frames_iterator() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")

    iterator = sess.audio_frames()

    async def consume() -> int:
        n = 0
        async for _ in iterator:
            n += 1
        return n

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)  # let consume block on the queue
    await sess.leave()
    assert await asyncio.wait_for(task, timeout=1.0) == 0


@pytest.mark.asyncio
async def test_inbound_overflow_drops_rather_than_blocking_sdk_thread() -> None:
    ch = InMemorySdkChannel()
    # Tiny buffer so we can saturate.
    sess = AliyunRtcSession(channel=ch, inbound_buffer_size=2)
    await sess.join("c-1", "tok", "engine-c-1")

    # 5 frames into a 2-slot buffer with no consumer attached. The
    # SdkChannel callback is synchronous; the engine wrapper must drop
    # excess rather than raise QueueFull back into the SDK thread.
    for i in range(5):
        ch.inject_remote_pcm("edge-c-1", b"\xaa\xbb", timestamp_ms=i)
    # Yield once so call_soon_threadsafe-scheduled puts run.
    await asyncio.sleep(0)

    iterator = sess.audio_frames()
    received: list[PcmFrame] = []
    for _ in range(2):
        received.append(await asyncio.wait_for(anext(iterator), timeout=1.0))
    assert len(received) == 2

    # Drain any further frames with timeout — should be empty after 2.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(iterator), timeout=0.05)

    await sess.leave()


# --------------------------------------------------------------------------
# Outbound flow + backpressure
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_audio_forwards_to_sdk_with_timestamp() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")

    await sess.push_audio(b"\x01\x02", timestamp_ms=10)
    await sess.push_audio(b"\x03\x04", timestamp_ms=20)
    assert ch.pushed == [(b"\x01\x02", 10), (b"\x03\x04", 20)]

    await sess.leave()


@pytest.mark.asyncio
async def test_buffer_full_signal_blocks_subsequent_push_until_drained() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")

    # First push succeeds.
    await sess.push_audio(b"\x01\x02", timestamp_ms=0)
    # SDK signals buffer-full; the NEXT push should block until drained.
    ch.signal_buffer_full()
    await asyncio.sleep(0)  # let drain_event.clear run

    push_task = asyncio.create_task(sess.push_audio(b"\x03\x04", timestamp_ms=1))
    # Should NOT complete before drain signal.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(push_task), timeout=0.05)
    assert push_task.done() is False

    # Drain → push completes.
    ch.signal_buffer_drained()
    await asyncio.wait_for(push_task, timeout=1.0)
    assert ch.pushed[-1] == (b"\x03\x04", 1)

    await sess.leave()


@pytest.mark.asyncio
async def test_push_audio_rejection_clears_drain_event_for_next_call() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")

    # Force the SDK's push_audio return to signal buffer-full WITHOUT
    # invoking the callback. The push still records the attempt; the
    # next push must block.
    ch.reject_next_push = True
    await sess.push_audio(b"\x01\x02", timestamp_ms=0)
    # That push was rejected — nothing in pushed[] (test double's choice).
    assert ch.pushed == []

    # Next push should now block on drain.
    push_task = asyncio.create_task(sess.push_audio(b"\x03\x04", timestamp_ms=1))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(push_task), timeout=0.05)

    ch.signal_buffer_drained()
    await asyncio.wait_for(push_task, timeout=1.0)
    assert ch.pushed == [(b"\x03\x04", 1)]

    await sess.leave()


@pytest.mark.asyncio
async def test_leave_releases_pending_push_waiters() -> None:
    ch = InMemorySdkChannel()
    sess = AliyunRtcSession(channel=ch)
    await sess.join("c-1", "tok", "engine-c-1")

    ch.signal_buffer_full()
    await asyncio.sleep(0)

    push_task = asyncio.create_task(sess.push_audio(b"\xaa\xbb", timestamp_ms=0))
    await asyncio.sleep(0.01)
    assert push_task.done() is False

    # Leave should release the wait, then push_audio will see
    # is_joined=False on the way back from the wait and... actually no:
    # the current implementation calls SDK.push regardless once drain
    # unblocks. The expected behaviour here is leave() sets drain_event,
    # the push resumes, the SDK is no longer joined, and it should be
    # a clean no-op (test double's _joined=False raises). We document
    # this as: leave() must not deadlock pending pushes; the push may
    # then fail loudly which the engine session catches.
    await sess.leave()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(push_task, timeout=1.0)
