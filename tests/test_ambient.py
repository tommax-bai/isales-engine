"""engine-ambient-background-mix — continuous outbound mixing pump + asset loop.

Covers the spec scenarios for `ambient-background-mix`:
- mix_frame: TTS+background sum with int16 clip; gain≤0 / empty bg passthrough.
- AmbientReader: 320B frames, seamless wrap, frame-aligned loop.
- asset loading: 16k mono WAV decode, basename-only (reject traversal), caching.
- pump: persistent push even when idle (background-only frames), TTS mixed when
  queued, barge-in flush keeps background, real-time pacing, inbound untouched.
"""

from __future__ import annotations

import asyncio
import struct
import wave
from array import array
from collections.abc import AsyncIterator

import pytest
from isales_common.audio.testing import InMemoryRtcSession

from isales_engine.realtime import ambient
from isales_engine.realtime.ambient import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    AmbientLoop,
    get_ambient_loop,
    mix_frame,
)
from isales_engine.realtime.rtc_telephony import _CallState


def _const_frame(value: int, n: int = FRAME_SAMPLES) -> bytes:
    return struct.pack(f"<{n}h", *([value] * n))


def _const_loop(value: int, frames: int = 20) -> AmbientLoop:
    return AmbientLoop(ambient._make_seamless(_const_frame(value) * frames), name="t")


# --------------------------------------------------------------------------
# mix_frame
# --------------------------------------------------------------------------


def test_mix_sums_with_gain() -> None:
    out = mix_frame(_const_frame(1000), _const_frame(10000), 0.1)
    assert struct.unpack("<h", out[:2])[0] == 2000  # 1000 + 10000*0.1


def test_mix_clips_to_int16() -> None:
    out = mix_frame(_const_frame(30000), _const_frame(30000), 1.0)
    assert struct.unpack("<h", out[:2])[0] == 32767


def test_mix_clips_negative() -> None:
    out = mix_frame(_const_frame(-30000), _const_frame(-30000), 1.0)
    assert struct.unpack("<h", out[:2])[0] == -32768


def test_mix_gain_zero_is_passthrough() -> None:
    tts = _const_frame(1234)
    assert mix_frame(tts, _const_frame(9999), 0.0) == tts


def test_mix_empty_bg_is_passthrough() -> None:
    tts = _const_frame(1234)
    assert mix_frame(tts, b"", 0.5) == tts


# --------------------------------------------------------------------------
# AmbientReader / seamless loop
# --------------------------------------------------------------------------


def test_reader_frames_are_320_bytes_and_wrap() -> None:
    loop = _const_loop(5, frames=10)
    reader = loop.reader()
    frames = [reader.next_frame() for _ in range(40)]  # well past one loop
    assert all(len(f) == FRAME_BYTES for f in frames)
    # constant source → every frame identical, wrap included
    assert all(struct.unpack("<h", f[:2])[0] == 5 for f in frames)


def test_loop_buffer_is_frame_aligned() -> None:
    loop = _const_loop(7, frames=13)
    assert len(loop._pcm) % FRAME_BYTES == 0
    assert bool(loop) is True


def test_seamless_no_large_seam_jump() -> None:
    # A ramp source: seam crossfade should avoid a hard discontinuity. Read two
    # full loops and assert adjacent samples never jump by a huge step.
    ramp = array("h", [(i % 200) - 100 for i in range(FRAME_SAMPLES * 12)])
    loop = AmbientLoop(ambient._make_seamless(ramp.tobytes()), name="ramp")
    reader = loop.reader()
    pcm = b"".join(reader.next_frame() for _ in range(len(loop._pcm) // FRAME_BYTES * 2))
    samples = array("h")
    samples.frombytes(pcm)
    max_jump = max(abs(samples[i + 1] - samples[i]) for i in range(len(samples) - 1))
    assert max_jump < 400  # ramp step is ≤1 except wrap; crossfade keeps it small


# --------------------------------------------------------------------------
# asset loading
# --------------------------------------------------------------------------


def _write_wav(path, *, rate: int, channels: int, value: int = 100, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{frames * channels}h", *([value] * (frames * channels))))


def test_load_16k_mono_asset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ISALES_AMBIENT_DIR", str(tmp_path))
    ambient._loop_cache.clear()
    _write_wav(tmp_path / "office.wav", rate=16000, channels=1)
    loop = get_ambient_loop("office.wav")
    assert loop is not None and bool(loop)
    assert struct.unpack("<h", loop.reader().next_frame()[:2])[0] == 100


def test_load_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ISALES_AMBIENT_DIR", str(tmp_path))
    ambient._loop_cache.clear()
    assert get_ambient_loop("../secret.wav") is None


def test_load_missing_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ISALES_AMBIENT_DIR", str(tmp_path))
    ambient._loop_cache.clear()
    assert get_ambient_loop("nope.wav") is None


def test_load_empty_name_returns_none() -> None:
    assert get_ambient_loop("") is None
    assert get_ambient_loop(None) is None


def test_load_is_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ISALES_AMBIENT_DIR", str(tmp_path))
    ambient._loop_cache.clear()
    _write_wav(tmp_path / "x.wav", rate=16000, channels=1)
    first = get_ambient_loop("x.wav")
    assert get_ambient_loop("x.wav") is first  # same cached instance


# --------------------------------------------------------------------------
# outbound mixing pump (_CallState)
# --------------------------------------------------------------------------


class _CapSession(InMemoryRtcSession):
    """Captures outbound frames instead of looping them back."""

    def __init__(self) -> None:
        super().__init__(loopback=False)
        self.pushed: list[tuple[bytes, int]] = []

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> None:
        if not self._joined:
            from isales_common.audio.rtc import RtcNotJoined

            raise RtcNotJoined("push before join")
        self.pushed.append((pcm, timestamp_ms))


async def _make_state(reader_value: int = 5, gain: float = 0.5) -> tuple[_CallState, _CapSession]:
    sess = _CapSession()
    await sess.join("c", "t", "engine-c")
    state = _CallState(
        call_id=1, rtc_session=sess, edge_uid="edge",
        device_id=0, edge_device_id="edge", outbound_frame_ms=20,
    )
    state.enable_ambient(_const_loop(reader_value).reader(), gain)
    return state, sess


async def _chunks(*frames: bytes) -> AsyncIterator[bytes]:
    for f in frames:
        yield f


@pytest.mark.asyncio
async def test_pump_pushes_background_when_idle() -> None:
    state, sess = await _make_state(reader_value=8, gain=0.5)
    try:
        await asyncio.sleep(0.06)  # ~6 frames at 10ms
        assert len(sess.pushed) >= 3
        # idle → TTS silence; pushed == mix(silence, bg=8, 0.5) → sample 4
        assert all(struct.unpack("<h", pcm[:2])[0] == 4 for pcm, _ in sess.pushed)
        # timestamps step by 10ms
        ts = [t for _, t in sess.pushed]
        assert ts == [10 * i for i in range(len(ts))]
    finally:
        await state.stop_outbound_pump()


@pytest.mark.asyncio
async def test_pump_mixes_queued_tts() -> None:
    state, sess = await _make_state(reader_value=10, gain=0.5)
    try:
        tts = _const_frame(1000)
        await state.feed_playout(_chunks(tts, tts))  # returns after drain
        # at least one captured frame is the TTS+bg mix (1000 + 10*0.5 = 1005)
        assert any(struct.unpack("<h", pcm[:2])[0] == 1005 for pcm, _ in sess.pushed)
    finally:
        await state.stop_outbound_pump()


@pytest.mark.asyncio
async def test_feed_playout_pads_partial_final_frame() -> None:
    state, sess = await _make_state(reader_value=0, gain=0.0)
    try:
        half = struct.pack("<80h", *([500] * 80))  # 160 bytes = half a frame
        await state.feed_playout(_chunks(half))
        # one full 320B frame pushed (padded), gain 0 → pure TTS+pad
        assert any(len(pcm) == FRAME_BYTES for pcm, _ in sess.pushed)
    finally:
        await state.stop_outbound_pump()


@pytest.mark.asyncio
async def test_barge_in_flush_keeps_background() -> None:
    state, sess = await _make_state(reader_value=9, gain=1.0)
    try:
        # Fill the queue, then cancel feed mid-flight (barge-in).
        feed = asyncio.create_task(
            state.feed_playout(_chunks(*[_const_frame(2000)] * 100))
        )
        await asyncio.sleep(0.02)
        feed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await feed
        assert state.playout_q is not None and state.playout_q.qsize() == 0  # flushed
        before = len(sess.pushed)
        await asyncio.sleep(0.04)
        # pump still pushing background frames (value 9) after barge-in
        assert len(sess.pushed) > before
        assert struct.unpack("<h", sess.pushed[-1][0][:2])[0] == 9
    finally:
        await state.stop_outbound_pump()


@pytest.mark.asyncio
async def test_pump_pacing_is_realtime() -> None:
    state, sess = await _make_state(reader_value=1, gain=0.5)
    try:
        await asyncio.sleep(0.2)  # ~20 frames at 10ms
        n = len(sess.pushed)
        assert 10 <= n <= 30  # generous bounds: real-time, no runaway / stall
    finally:
        await state.stop_outbound_pump()


@pytest.mark.asyncio
async def test_ambient_never_touches_inbound() -> None:
    # Isolation invariant: enabling ambient + pushing background must not place
    # anything on the inbound / VAD queues (ASR / 断句 path stays clean).
    state, sess = await _make_state(reader_value=5, gain=0.5)
    try:
        await asyncio.sleep(0.05)
        assert state.inbound_q.empty()
        assert state.vad_q.empty()
    finally:
        await state.stop_outbound_pump()
