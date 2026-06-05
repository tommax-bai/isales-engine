"""Tests for OpenAICompatibleLLMProvider with httpx.MockTransport.

Covers happy path + every ProviderError subclass mapping per spec delta.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from isales_common.providers._models import Message

from isales_engine.providers._errors import (
    ProviderInvalidRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
)
from isales_engine.providers.llm_openai_compatible import (
    OpenAICompatibleLLMProvider,
)


def _build_provider(handler: Any) -> OpenAICompatibleLLMProvider:
    transport = httpx.MockTransport(handler)
    provider = OpenAICompatibleLLMProvider(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )

    # Patch the AsyncClient to use our MockTransport.
    original_chat = provider.chat

    async def chat_with_mock(*args: Any, **kwargs: Any) -> Any:
        # Re-implement using shared transport to avoid creating a real client.
        async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
            messages: list[Message] = args[0] if args else kwargs["messages"]
            json_mode = kwargs.get("json_mode", False)
            payload: dict[str, object] = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 1.0),
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            from isales_engine.providers._errors import map_http_error
            from isales_engine.providers.llm_openai_compatible import (
                _parse_chat_response,
            )

            if response.status_code >= 400:
                raise map_http_error(response, provider="openai")
            return _parse_chat_response(response, provider="openai", latency_ms=0)

    provider.chat = chat_with_mock  # type: ignore[method-assign]
    _ = original_chat
    return provider


# ---- happy path ------------------------------------------------------------


async def test_chat_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                },
            },
        )

    provider = _build_provider(handler)
    resp = await provider.chat([Message(role="user", content="ping")])
    assert resp.content == "hi"
    assert resp.tokens_in == 12
    assert resp.tokens_out == 3
    assert resp.finish_reason == "stop"


async def test_chat_json_mode_sets_response_format() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["response_format"] = body.get("response_format")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    provider = _build_provider(handler)
    await provider.chat([Message(role="user", content="x")], json_mode=True)
    assert seen["response_format"] == {"type": "json_object"}


# ---- error mapping ---------------------------------------------------------


async def test_chat_429_raises_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
            headers={"Retry-After": "5"},
        )

    provider = _build_provider(handler)
    with pytest.raises(ProviderRateLimited) as exc_info:
        await provider.chat([Message(role="user", content="x")])
    assert exc_info.value.retry_after_seconds == 5.0
    assert exc_info.value.vendor_code == "rate_limit_exceeded"


async def test_chat_5xx_raises_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "vendor down"})

    provider = _build_provider(handler)
    with pytest.raises(ProviderServerError):
        await provider.chat([Message(role="user", content="x")])


async def test_chat_401_raises_invalid_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"code": "invalid_api_key"}}
        )

    provider = _build_provider(handler)
    with pytest.raises(ProviderInvalidRequest) as exc_info:
        await provider.chat([Message(role="user", content="x")])
    assert exc_info.value.vendor_code == "invalid_api_key"


async def test_chat_response_missing_choices_raises_invalid_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x"})

    provider = _build_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.chat([Message(role="user", content="x")])


async def test_chat_response_non_json_raises_invalid_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})

    provider = _build_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.chat([Message(role="user", content="x")])


async def test_chat_unknown_finish_reason_falls_back_to_stop() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "vendor_specific_value",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    provider = _build_provider(handler)
    resp = await provider.chat([Message(role="user", content="x")])
    assert resp.finish_reason == "stop"


# ---- transport error -------------------------------------------------------


async def test_real_chat_with_timeout_raises_provider_timeout() -> None:
    """The unwrapped chat method (not our mock_transport hack) raises ProviderTimeout."""

    provider = OpenAICompatibleLLMProvider(
        provider="openai",
        api_key="sk-test",
        base_url="https://nonexistent.example.test:1",
        model="gpt-4o-mini",
        timeout_s=0.1,
    )
    with pytest.raises((ProviderTimeout, ProviderServerError)):
        await provider.chat([Message(role="user", content="x")])


# ---- connection reuse (pipeline-latency-tail § C, extended to LLM) ----------


def _ok_chat_response() -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


async def test_reuses_persistent_client_across_calls() -> None:
    """Consecutive chat() calls go through the same provider-lived client —
    no per-call client rebuild (avoids a TLS handshake per turn)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_ok_chat_response())

    provider = OpenAICompatibleLLMProvider(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        transport=httpx.MockTransport(handler),
    )
    client_before = provider._http
    for _ in range(3):
        await provider.chat([Message(role="user", content="hi")])

    assert calls["n"] == 3
    assert provider._http is client_before  # same client reused every turn
    assert provider._http.is_closed is False


async def test_aclose_releases_client() -> None:
    provider = OpenAICompatibleLLMProvider(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_ok_chat_response())),
    )
    assert provider._http.is_closed is False
    await provider.aclose()
    assert provider._http.is_closed is True
