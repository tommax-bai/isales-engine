"""Unit tests for the producer/consumer + pre-synthesis playback pipeline
(pipeline-latency-tail § B — ``run_loop._play_streaming`` / ``_SynthJob``).

Uses a lightweight fake PipelineStream (only ``.sentences()`` / ``.result``
/ ``.turn_id`` are touched) plus a recording TTS and a gated telephony mock
so the concurrency can be asserted deterministically (no wall-clock races).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from isales_common.providers.tts import TTSProvider

from isales_engine.call_session import CallSession
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import _play_streaming


def _make_session(call_id: int = 1) -> CallSession:
    return CallSession(
        call_record_id=call_id,
        campaign_id=10,
        lead_id=5,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


class _FakeStream:
    """Minimal stand-in for PipelineStream used by _play_streaming."""

    def __init__(self, sentences: list[str], *, turn_id: int = 1) -> None:
        self._sentences = sentences
        self.turn_id = turn_id
        self.result = SimpleNamespace(error=None, reply_text="")
        self.referee_task = None

    async def sentences(self) -> AsyncIterator[str]:
        for s in self._sentences:
            self.result.reply_text += s
            yield s


class _RecordingTTS(TTSProvider):
    """Records synth start order + tracks open generators (leak detection)."""

    def __init__(
        self, *, synth_delay_s: float = 0.0, chunks_per_sentence: int = 1
    ) -> None:
        self.synth_starts: list[str] = []
        self.closed: list[str] = []
        self.active = 0
        self._synth_delay_s = synth_delay_s
        self._chunks_per_sentence = chunks_per_sentence
        self.fail_on: set[str] = set()

    async def synthesize_stream(
        self, text: str, voice_id: str
    ) -> AsyncIterator[bytes]:
        self.synth_starts.append(text)
        self.active += 1
        try:
            if text in self.fail_on:
                raise RuntimeError(f"synth boom: {text}")
            for i in range(self._chunks_per_sentence):
                if self._synth_delay_s:
                    await asyncio.sleep(self._synth_delay_s)
                yield f"{text}#{i}".encode()
        finally:
            self.active -= 1
            self.closed.append(text)


class _GatedTelephony(MockTelephonyClient):
    """Holds playback of the first chunk until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.first_chunk_played = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()  # default: don't hold (overridden per test)

    async def audio_out(self, call_id: int, chunks: AsyncIterator[bytes]) -> None:
        self._ensure_call(call_id)
        async for chunk in chunks:
            if not self.first_chunk_played.is_set():
                self.first_chunk_played.set()
                await self.release.wait()
            self.outbound_log[call_id].append(chunk)


async def _connected(tel: MockTelephonyClient, call_id: int) -> None:
    await tel.dial(call_id, "+x")
    async for _ in tel.events(call_id):
        break


# ---- happy path: ordered playback + first_audio_ms -------------------------


async def test_plays_all_sentences_in_order() -> None:
    session = _make_session()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(chunks_per_sentence=2)
    stream = _FakeStream(["你好。", "今天天气不错。", "再见。"])

    first_audio_ms, played = await _play_streaming(
        session, tel, tts, stream,  # type: ignore[arg-type]
        voice_id="v", filler=None, processing_start=time.monotonic(),
    )

    assert played is True
    assert first_audio_ms is not None and first_audio_ms >= 0
    got = [c.decode() for c in tel.outbound_log[session.call_record_id]]
    assert got == [
        "你好。#0", "你好。#1",
        "今天天气不错。#0", "今天天气不错。#1",
        "再见。#0", "再见。#1",
    ]
    assert tts.active == 0  # every synth generator closed


# ---- pre-synthesis: N+1 synth starts while N is still playing ---------------


async def test_pre_synthesizes_next_sentence_during_playback() -> None:
    session = _make_session()
    tel = _GatedTelephony()
    tel.release.clear()  # hold the first chunk so sentence 0 stays "playing"
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(chunks_per_sentence=1)
    stream = _FakeStream(["第一句。", "第二句。", "第三句。"])

    task = asyncio.create_task(
        _play_streaming(
            session, tel, tts, stream,  # type: ignore[arg-type]
            voice_id="v", filler=None, processing_start=time.monotonic(),
        )
    )
    # Sentence 0's first chunk is now being played (held by the gate).
    await asyncio.wait_for(tel.first_chunk_played.wait(), timeout=1.0)
    # Give the producer a slice to pre-synthesize ahead (maxsize=2).
    await asyncio.sleep(0.05)
    # Sentence 1 (and 2, bounded by maxsize) synthesis MUST have started while
    # sentence 0 is still playing — no per-sentence first-byte gap.
    assert "第二句。" in tts.synth_starts

    tel.release.set()
    first_audio_ms, played = await asyncio.wait_for(task, timeout=2.0)
    assert played is True
    assert first_audio_ms is not None
    assert [c.decode() for c in tel.outbound_log[session.call_record_id]] == [
        "第一句。#0", "第二句。#0", "第三句。#0",
    ]


# ---- barge-in three states: cancel current play, clean up the pipeline ------


async def _run_until_first_chunk(
    session: CallSession, tel: _GatedTelephony, tts: _RecordingTTS, stream: _FakeStream
) -> asyncio.Task[Any]:
    task = asyncio.create_task(
        _play_streaming(
            session, tel, tts, stream,  # type: ignore[arg-type]
            voice_id="v", filler=None, processing_start=time.monotonic(),
        )
    )
    await asyncio.wait_for(tel.first_chunk_played.wait(), timeout=1.0)
    return task


async def test_bargein_while_playing_returns_false_and_cleans_up() -> None:
    """Barge-in lands while sentence 0 is mid-play; next sentences are still
    being pre-synthesized. Cancelling current_speaking_task must yield
    fully_played=False and close every in-flight synth (no leak)."""
    session = _make_session()
    tel = _GatedTelephony()
    tel.release.clear()
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(synth_delay_s=0.02, chunks_per_sentence=3)
    stream = _FakeStream(["第一句。", "第二句。", "第三句。"])

    task = await _run_until_first_chunk(session, tel, tts, stream)
    assert session.current_speaking_task is not None
    # Barge-in: partial monitor cancels the playing audio_out task.
    session.current_speaking_task.cancel()
    tel.release.set()  # unblock the (now-cancelled) playback

    first_audio_ms, played = await asyncio.wait_for(task, timeout=2.0)
    assert played is False
    assert first_audio_ms is not None
    # All synth generators (current + any pre-synthesized) were closed.
    assert tts.active == 0


async def test_bargein_with_jobs_queued_closes_pending_synth() -> None:
    """Barge-in after the producer has filled the bounded queue: the queued
    (ready) jobs must be drained + closed, not leaked."""
    session = _make_session()
    tel = _GatedTelephony()
    tel.release.clear()
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(chunks_per_sentence=1)
    stream = _FakeStream(["第一句。", "第二句。", "第三句。", "第四句。"])

    task = await _run_until_first_chunk(session, tel, tts, stream)
    # Let the producer pre-synthesize + fill the maxsize=2 queue.
    await asyncio.sleep(0.05)
    assert session.current_speaking_task is not None
    session.current_speaking_task.cancel()
    tel.release.set()

    _first, played = await asyncio.wait_for(task, timeout=2.0)
    assert played is False
    assert tts.active == 0  # queued jobs closed, nothing left synthesizing


# ---- exception injection: TTS synth failure does not crash the call --------


async def test_tts_synth_error_does_not_crash_and_records_error() -> None:
    session = _make_session()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(chunks_per_sentence=1)
    tts.fail_on = {"第二句。"}
    stream = _FakeStream(["第一句。", "第二句。", "第三句。"])

    first_audio_ms, played = await _play_streaming(
        session, tel, tts, stream,  # type: ignore[arg-type]
        voice_id="v", filler=None, processing_start=time.monotonic(),
    )

    # First sentence played; the failing sentence broke the turn gracefully.
    assert first_audio_ms is not None
    assert played is True  # not treated as a user interruption
    assert stream.result.error is not None and "tts_playback" in stream.result.error
    got = [c.decode() for c in tel.outbound_log[session.call_record_id]]
    assert got[0] == "第一句。#0"
    assert tts.active == 0


async def test_producer_stream_error_is_isolated() -> None:
    """Producer-side (sentence generator) raising mid-stream must not hang
    the consumer — sentences already pre-synthesized still play."""

    class _BoomStream(_FakeStream):
        async def sentences(self) -> AsyncIterator[str]:
            yield "第一句。"
            raise RuntimeError("main stream boom")

    session = _make_session()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(chunks_per_sentence=1)
    stream = _BoomStream(["unused"])

    first_audio_ms, played = await asyncio.wait_for(
        _play_streaming(
            session, tel, tts, stream,  # type: ignore[arg-type]
            voice_id="v", filler=None, processing_start=time.monotonic(),
        ),
        timeout=2.0,
    )
    # The one yielded sentence played; producer error didn't crash _play_streaming.
    assert played is True
    assert first_audio_ms is not None
    assert [c.decode() for c in tel.outbound_log[session.call_record_id]] == ["第一句。#0"]
    assert tts.active == 0
