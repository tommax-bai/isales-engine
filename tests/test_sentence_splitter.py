"""Tests for streaming/sentence_splitter (pipeline-stream-and-referee)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from isales_engine.streaming.sentence_splitter import (
    MAX_SENTENCE_CHARS,
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
async def test_char_cap_forces_split():
    long = "啊" * (MAX_SENTENCE_CHARS + 10)  # no terminator at all
    out = await _collect(split_sentences(_stream(long)))
    # First chunk is exactly the cap; remainder flushed at stream end.
    assert out[0] == "啊" * MAX_SENTENCE_CHARS
    assert "".join(out) == long


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
