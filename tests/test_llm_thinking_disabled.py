"""Tests for default thinking/推理-off injection on the OpenAI-compatible LLM
provider (engine-llm-disable-thinking).

provider-abc § Requirement: LLM 思考/推理模式默认关闭 — every chat / chat_stream
request MUST carry the provider-specific disable-thinking field so no role
inherits the vendor default (DashScope qwen3.6-flash / Volcengine doubao-seed-1.6
default thinking ON → 上千 reasoning token → 10-15s/turn first audio).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from isales_common.providers._models import Message

from isales_engine.providers.llm_openai_compatible import (
    OpenAICompatibleLLMProvider,
)


def _ok_chat_response() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _sse(*chunks: str) -> bytes:
    return ("".join(f"data: {c}\n\n" for c in chunks)).encode()


def _capture_chat_provider(
    provider_name: str, seen: dict[str, Any]
) -> OpenAICompatibleLLMProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_chat_response())

    return OpenAICompatibleLLMProvider(
        provider=provider_name,
        api_key="sk-test",
        base_url="https://vendor.example/v1",
        model="some-model",
        transport=httpx.MockTransport(handler),
    )


def _capture_stream_provider(
    provider_name: str, seen: dict[str, Any]
) -> OpenAICompatibleLLMProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse(
                '{"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}',
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    return OpenAICompatibleLLMProvider(
        provider=provider_name,
        api_key="sk-test",
        base_url="https://vendor.example/v1",
        model="some-model",
        transport=httpx.MockTransport(handler),
    )


# ---- chat() (referee / extractor / restructure / main fallback) ------------


async def test_chat_dashscope_disables_thinking() -> None:
    seen: dict[str, Any] = {}
    provider = _capture_chat_provider("dashscope", seen)
    await provider.chat([Message(role="user", content="x")])
    assert seen["enable_thinking"] is False


async def test_chat_volcengine_disables_thinking() -> None:
    seen: dict[str, Any] = {}
    provider = _capture_chat_provider("volcengine", seen)
    await provider.chat([Message(role="user", content="x")])
    assert seen["thinking"] == {"type": "disabled"}


async def test_chat_unknown_provider_injects_nothing() -> None:
    seen: dict[str, Any] = {}
    provider = _capture_chat_provider("openai", seen)
    await provider.chat([Message(role="user", content="x")])
    assert "enable_thinking" not in seen
    assert "thinking" not in seen


# ---- chat_stream() (main) --------------------------------------------------


async def test_chat_stream_dashscope_disables_thinking() -> None:
    seen: dict[str, Any] = {}
    provider = _capture_stream_provider("dashscope", seen)
    async for _ in provider.chat_stream([Message(role="user", content="x")]):
        pass
    assert seen["enable_thinking"] is False
    # coexists with streaming params
    assert seen["stream"] is True


async def test_chat_stream_volcengine_disables_thinking() -> None:
    seen: dict[str, Any] = {}
    provider = _capture_stream_provider("volcengine", seen)
    async for _ in provider.chat_stream([Message(role="user", content="x")]):
        pass
    assert seen["thinking"] == {"type": "disabled"}
    assert seen["stream"] is True
