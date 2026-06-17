"""Tests for streaming/sentence_splitter (pipeline-stream-and-referee)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from isales_engine.streaming.sentence_splitter import (
    MAX_SENTENCE_CHARS,
    SOFT_SENTENCE_CHARS,
    split_sentences,
)


async def _stream(*parts: str) -> AsyncIterator[str]:
    for p in parts:
        yield p


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    return [s async for s in stream]


@pytest.mark.asyncio
async def test_chinese_punctuation_splits():
    out = await _collect(split_sentences(_stream("您好。", "请问现在方便吗？", "好的！")))
    assert out == ["您好。", "请问现在方便吗？", "好的！"]


@pytest.mark.asyncio
async def test_fragmented_tokens_buffer_across_boundaries():
    # "您"+"好"+"。" arrive as three separate tokens.
    out = await _collect(split_sentences(_stream("您", "好", "。", "在", "吗", "？")))
    assert out == ["您好。", "在吗？"]


@pytest.mark.asyncio
async def test_double_newline_splits():
    out = await _collect(split_sentences(_stream("第一段", "\n\n", "第二段")))
    assert out == ["第一段", "第二段"]


@pytest.mark.asyncio
async def test_no_punctuation_run_under_ceiling_stays_whole():
    # engine-sentence-splitter-soft-boundary: a punctuation-less run that
    # exceeds the soft threshold but stays under the hard ceiling is NOT cut
    # mid-word anymore (the old behavior hard-cut at 50). It flushes whole.
    run = "啊" * (SOFT_SENTENCE_CHARS + 10)  # past soft, no boundary char, < ceiling
    out = await _collect(split_sentences(_stream(run)))
    assert out == [run]


@pytest.mark.asyncio
async def test_hard_ceiling_forces_split_for_punctuationless_runon():
    # Only a punctuation-less run reaching the hard ceiling falls back to a
    # length cut (bounds first-audio latency for malformed output).
    run = "啊" * (MAX_SENTENCE_CHARS + 10)  # no terminator / soft boundary at all
    out = await _collect(split_sentences(_stream(run)))
    assert out[0] == "啊" * MAX_SENTENCE_CHARS  # cut at the ceiling
    assert "".join(out) == run  # remainder flushed at stream end


@pytest.mark.asyncio
async def test_long_sentence_splits_at_soft_boundary_not_mid_word():
    # Past the soft threshold, the cut lands on the next pause punctuation
    # (a natural clause break), keeping that punctuation in the emitted unit —
    # never a mid-word hard cut.
    long_clause = "啊" * SOFT_SENTENCE_CHARS  # reaches the soft threshold
    out = await _collect(split_sentences(_stream(long_clause + "，", "好的。")))
    assert out == [long_clause + "，", "好的。"]


@pytest.mark.asyncio
async def test_comma_below_soft_threshold_does_not_split():
    # A short sentence with internal commas is untouched by length — it only
    # splits on the real terminator (behavior unchanged from before).
    out = await _collect(split_sentences(_stream("您好张总，我是小何，方便聊两句吗？")))
    assert out == ["您好张总，我是小何，方便聊两句吗？"]


@pytest.mark.asyncio
async def test_flush_tail_without_terminator():
    out = await _collect(split_sentences(_stream("没有标点的尾巴")))
    assert out == ["没有标点的尾巴"]


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing():
    out = await _collect(split_sentences(_stream()))
    assert out == []


@pytest.mark.asyncio
async def test_empty_and_whitespace_tokens_skipped():
    out = await _collect(split_sentences(_stream("", "你好", "", "。", "   ")))
    assert out == ["你好。"]


@pytest.mark.asyncio
async def test_ascii_punctuation():
    out = await _collect(split_sentences(_stream("Hello.", " World?")))
    assert out == ["Hello.", "World?"]


# ---- engine-turn-latency-and-tts-guard: drop chunks with no synthesizable text
# (Volcengine TTS rejects punctuation-only text with code=45002001). -----------


@pytest.mark.asyncio
async def test_ellipsis_punctuation_only_chunks_dropped():
    # "呃...那个。" — the ellipsis splits on each "." into bare "." chunks the
    # vendor can't voice; chunks keeping a readable char survive (call 194).
    out = await _collect(split_sentences(_stream("呃...那个。")))
    assert out == ["呃.", "那个。"]


@pytest.mark.asyncio
async def test_all_punctuation_yields_nothing():
    out = await _collect(split_sentences(_stream("。。。", "！", "...", "   ")))
    assert out == []


@pytest.mark.asyncio
async def test_digit_is_synthesizable():
    out = await _collect(split_sentences(_stream("3。")))
    assert out == ["3。"]
