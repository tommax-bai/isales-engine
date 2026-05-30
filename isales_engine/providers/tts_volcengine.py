"""Volcengine streaming TTS provider.

Spec: provider-abc § Requirement: TTS Provider 异步流式合成接口;
      impl-engine-providers proposal § "真实 TTS Provider 实现".

Implements ``synthesize_stream(text, voice_id) -> AsyncIterator[bytes]``
over the publicly-documented Volcengine HTTP streaming TTS endpoint. We
chose a direct ``httpx.stream(...)`` integration over a vendor SDK for
three reasons:

1. The protocol is a single POST + chunked response — adding a SDK
   dependency for one HTTP call is heavyweight.
2. The ``ProviderError`` mapping needed is identical to the LLM provider's,
   so we can reuse ``_errors.map_http_error`` / ``map_transport_error``.
3. Switching to the official SDK later (v2 candidate) only touches this
   file; the ABC contract + caller code never changes.

**Vendor verification required before production**: protocol assumptions
in this file MUST be smoke-tested against a real Volcengine account before
deploying. Output format target is 8 kHz mono 16-bit LE PCM (matches GSM
modem-controller stage-6 path); endpoint + voice id list per Volcengine
console.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

import httpx
from isales_common.providers.tts import TTSProvider

from isales_engine.providers._errors import (
    map_http_error,
    map_transport_error,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0


class VolcengineTTSProvider(TTSProvider):
    """Streaming TTS over the Volcengine HTTP API.

    The provider performs a POST with the synthesis request and yields PCM
    chunks as they arrive. Failures map onto the standard ``ProviderError``
    hierarchy (same rules as the LLM provider). Per impl-engine-providers
    design § 3, transient (5xx / timeout) failures are retried once before
    surfacing — the user impact of a TTS failure is "user hears silence
    where the AI reply should have been," so an aggressive retry would
    only stretch that silence.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        app_key: str,
        app_token: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._app_key = app_key
        self._app_token = app_token
        self._timeout_s = timeout_s
        # Reuse a single AsyncClient across requests so the connection pool
        # (TCP + TLS) is kept warm — avoids per-call handshake latency.
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    async def aclose(self) -> None:
        """Close the shared AsyncClient (call on provider shutdown)."""

        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def close(self) -> None:
        """Alias for :meth:`aclose` — lifecycle close on provider shutdown."""

        await self.aclose()

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
    ) -> AsyncIterator[bytes]:
        last_error: Exception | None = None
        # One initial attempt + 1 retry on transient errors.
        for attempt in range(2):
            try:
                async for chunk in self._stream_once(text, voice_id):
                    yield chunk
                return
            except httpx.HTTPError as exc:
                last_error = map_transport_error(exc, provider="volcengine_tts")
                if attempt == 0 and _is_retryable(last_error):
                    logger.warning(
                        "volcengine_tts_retrying after transport error: %s", exc
                    )
                    continue
                raise last_error from exc

        if last_error is not None:
            raise last_error

    async def _stream_once(
        self, text: str, voice_id: str
    ) -> AsyncIterator[bytes]:
        url = f"{self._endpoint}/tts"
        headers = {
            "Authorization": f"Bearer; {self._app_token}",
            "Content-Type": "application/json",
            "Resource-Id": "tts",
            "App-Key": self._app_key,
        }
        # Payload shape per Volcengine 流式 TTS public docs. Output PCM
        # 8 kHz mono 16-bit LE matches the modem-controller PCM channel.
        payload = {
            "audio": {
                "voice_type": voice_id,
                "encoding": "pcm",
                "rate": 8000,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "reqid": _request_id(),
                "text": text,
                "operation": "submit",
            },
        }

        start = time.monotonic()
        first_byte_logged = False

        client = self._get_client()
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                response = httpx.Response(
                    status_code=response.status_code,
                    content=body,
                    headers=response.headers,
                )
                raise map_http_error(response, provider="volcengine_tts")

            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if not first_byte_logged:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    logger.info(
                        "volcengine_tts_first_byte latency_ms=%s text_len=%s",
                        latency_ms,
                        len(text),
                    )
                    first_byte_logged = True
                yield chunk


def _is_retryable(error: Exception) -> bool:
    """Only transient errors (5xx / timeouts) get the one retry."""

    from isales_engine.providers._errors import (
        ProviderServerError,
        ProviderTimeout,
    )

    return isinstance(error, ProviderServerError | ProviderTimeout)


def _request_id() -> str:
    """Per-call unique id (ms-resolution clock is plenty for us)."""

    import uuid

    return uuid.uuid4().hex


# Stash the json import so unused-import linters don't strip it; we use it
# implicitly via httpx.json= kwarg on POST.
_ = json
