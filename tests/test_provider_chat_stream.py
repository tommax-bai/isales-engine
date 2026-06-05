"""Tests for LLMProvider.chat_stream (pipeline-stream-and-referee).

Covers the OpenAI-compatible SSE path (happy stream / mid-stream HTTP error /
transport error / malformed chunk skip / usage capture) and the mock provider's
simulated streaming.
"""

from __future__ import annotations

import httpx
import pytest
from isales_common.providers._models import Message

from isales_engine.providers._errors import ProviderRateLimited, ProviderTimeout
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.llm_openai_compatible import (
    _SSE_DONE,
    OpenAICompatibleLLMProvider,
    _parse_sse_line,
)


def _sse(*chunks: str) -> bytes:
    return ("".join(f"data: {c}\n\n" for c in chunks)).encode()


def _provider(handler) -> OpenAICompatibleLLMProvider:
    return OpenAICompatibleLLMProvider(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        transport=httpx.MockTransport(handler),
    )


def _delta(text: str) -> str:
    return f'{{"choices":[{{"delta":{{"content":"{text}"}},"finish_reason":null}}]}}'


async def _collect(provider: OpenAICompatibleLLMProvider) -> str:
    out = []
    async for tok in provider.chat_stream([Message(role="user", content="hi")]):
        out.append(tok)
    return "".join(out)


# ---- SSE line parser unit tests -------------------------------------------


def test_parse_sse_blank_and_non_data_lines_skip():
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": ping") is None
    assert _parse_sse_line("event: message") is None


def test_parse_sse_done_sentinel():
    assert _parse_sse_line("data: [DONE]") is _SSE_DONE


def test_parse_sse_delta():
    delta, finish, usage = _parse_sse_line(f"data: {_delta('你好')}")
    assert delta == "你好"
    assert finish is None
    assert usage is None


def test_parse_sse_malformed_json_skips():
    assert _parse_sse_line("data: {not json") is None


# ---- Happy streaming path --------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse(
            _delta("您好"),
            _delta("，请问"),
            _delta("现在方便吗？"),
            '{"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":42,"completion_tokens":7}}',
            "[DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = _provider(handler)
    text = await _collect(provider)
    assert text == "您好，请问现在方便吗？"
    assert provider.last_call_tokens_in == 42
    assert provider.last_call_tokens_out == 7
    assert provider.last_call_finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_stream_malformed_chunk_does_not_abort():
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse(_delta("abc"), "{broken", _delta("def"), "[DONE]")
        return httpx.Response(200, content=body)

    text = await _collect(_provider(handler))
    assert text == "abcdef"


# ---- Error mapping ---------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_http_error_maps_before_first_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"code": "RPM_EXCEEDED"})

    with pytest.raises(ProviderRateLimited):
        await _collect(_provider(handler))


@pytest.mark.asyncio
async def test_chat_stream_transport_error_maps():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    with pytest.raises(ProviderTimeout):
        await _collect(_provider(handler))


# ---- Mock provider streaming ----------------------------------------------


@pytest.mark.asyncio
async def test_mock_chat_stream_yields_decided_content():
    mock = KeywordDrivenMockLLM()
    out = []
    async for tok in mock.chat_stream([Message(role="user", content="你好")]):
        out.append(tok)
    joined = "".join(out)
    # Default role reply (json since the mock still emits JSON pre-section-8).
    assert "好的" in joined
    assert mock.last_call_finish_reason == "stop"
    assert mock.calls  # call recorded


@pytest.mark.asyncio
async def test_mock_chat_stream_respects_timing_params():
    # Non-zero timing should still complete; we only assert it streams.
    mock = KeywordDrivenMockLLM(first_token_ms=1.0, per_token_ms=0.1)
    out = [tok async for tok in mock.chat_stream([Message(role="user", content="hi")])]
    assert out
