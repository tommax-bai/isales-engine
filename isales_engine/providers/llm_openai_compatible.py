"""OpenAI-compatible LLM provider (used for OpenAI + 火山豆包 + 兼容服务).

Spec: provider-abc § Requirement: LLM Provider chat 接口与 JSON Mode;
      role-prompt § Requirement: JSON Mode 强制策略.

Both OpenAI and 火山豆包 expose a compatible Chat Completions endpoint
(``POST {base_url}/chat/completions``) with ``response_format={"type":
"json_object"}`` for native JSON Mode. We wrap a single httpx call with the
``ProviderError`` mapping from ``providers/_errors.py``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

import httpx
from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider

from isales_engine.providers._errors import (
    as_provider_invalid_response,
    map_http_error,
    map_transport_error,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0


class OpenAICompatibleLLMProvider(LLMProvider):
    """Implements LLMProvider against any OpenAI-compatible chat-completions API.

    Used for:
      * OpenAI         — ``base_url=https://api.openai.com/v1``
      * Volcengine 豆包 — ``base_url=https://ark.cn-beijing.volces.com/api/v3``
      * Azure OpenAI / 第三方兼容 — ``base_url`` to the matching prefix

    The ``provider`` label is carried on every ``ProviderError`` so logs +
    runbooks can attribute failures.
    """

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._extra_headers = dict(extra_headers or {})
        # Optional injected transport (tests pass httpx.MockTransport). None →
        # the default real transport.
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport)

    async def chat(
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            async with self._client() as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, provider=self._provider) from exc

        latency_ms = int((time.monotonic() - start) * 1000)

        if response.status_code >= 400:
            raise map_http_error(response, provider=self._provider)

        return _parse_chat_response(
            response, provider=self._provider, latency_ms=latency_ms
        )

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream content deltas over SSE (``stream=true``).

        Yields each ``choices[0].delta.content`` chunk. ``stream_options.
        include_usage`` requests a trailing usage chunk so token counts land on
        ``last_call_tokens_in / _out`` without a second round trip. Errors are
        mapped to the ``ProviderError`` family, same as ``chat``.
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Reset usage metadata; populated from the final SSE chunk if present.
        self.last_call_tokens_in = None
        self.last_call_tokens_out = None
        self.last_call_finish_reason = None

        try:
            async with self._client() as client, client.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    # Materialize the body so map_http_error can inspect it.
                    await response.aread()
                    raise map_http_error(response, provider=self._provider)
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is _SSE_DONE:
                        break
                    if chunk is None:
                        continue
                    delta, finish_reason, usage = chunk
                    if usage is not None:
                        self.last_call_tokens_in = _coerce_int(
                            usage.get("prompt_tokens")
                        )
                        self.last_call_tokens_out = _coerce_int(
                            usage.get("completion_tokens")
                        )
                    if finish_reason is not None:
                        self.last_call_finish_reason = _normalize_finish_reason(
                            finish_reason
                        )
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise map_transport_error(exc, provider=self._provider) from exc


# Sentinel signalling the SSE ``[DONE]`` terminator.
_SSE_DONE = object()


def _parse_sse_line(
    line: str,
) -> object:
    """Parse one SSE line into ``(delta, finish_reason, usage)`` or a sentinel.

    Returns:
      * ``None`` for blank / non-``data:`` lines (skip).
      * ``_SSE_DONE`` for the ``data: [DONE]`` terminator.
      * ``(delta_str, finish_reason | None, usage_dict | None)`` otherwise.

    A malformed JSON payload is skipped (returns ``None``) rather than aborting
    the stream — a single bad chunk should not drop the whole reply.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return _SSE_DONE
    try:
        obj = json.loads(data)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None

    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None

    delta_text = ""
    finish_reason: str | None = None
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    delta_text = content
            fr = first.get("finish_reason")
            if isinstance(fr, str):
                finish_reason = fr
    return (delta_text, finish_reason, usage)


def _parse_chat_response(
    response: httpx.Response, *, provider: str, latency_ms: int
) -> LLMResponse:
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise as_provider_invalid_response(
            f"{provider} response was not JSON: {exc}", provider=provider
        ) from exc

    if not isinstance(data, dict):
        raise as_provider_invalid_response(
            f"{provider} response root was not an object: {type(data).__name__}",
            provider=provider,
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise as_provider_invalid_response(
            f"{provider} response missing 'choices'", provider=provider
        )

    first = choices[0]
    if not isinstance(first, dict):
        raise as_provider_invalid_response(
            f"{provider} choices[0] was not an object", provider=provider
        )

    message = first.get("message")
    if not isinstance(message, dict):
        raise as_provider_invalid_response(
            f"{provider} choices[0].message missing", provider=provider
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise as_provider_invalid_response(
            f"{provider} choices[0].message.content not a string", provider=provider
        )

    finish_reason = _normalize_finish_reason(first.get("finish_reason"))
    raw_usage = data.get("usage")
    usage: dict[str, object] = raw_usage if isinstance(raw_usage, dict) else {}
    tokens_in = _coerce_int(usage.get("prompt_tokens"))
    tokens_out = _coerce_int(usage.get("completion_tokens"))

    return LLMResponse(
        content=content,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
    )


def _coerce_int(value: object) -> int:
    """Best-effort conversion to int; return 0 on any failure."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _normalize_finish_reason(value: object) -> str:
    """Map vendor finish_reason values onto LLMResponse's typed Literal.

    Unknown values fall back to ``"stop"`` (the safe default — caller treats
    output as complete).
    """

    if value in ("stop", "length", "content_filter", "tool_calls", "error"):
        return str(value)
    return "stop"
