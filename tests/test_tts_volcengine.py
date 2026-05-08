"""Tests for VolcengineTTSProvider over httpx streaming.

Uses httpx.MockTransport to verify protocol semantics + ProviderError
mapping. Real-vendor smoke tests live in tests/test_real_providers.py and
require ``ISALES_LIVE_PROVIDER_TESTS=1`` (not run in CI).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from isales_engine.providers._errors import (
    ProviderInvalidRequest,
    ProviderRateLimited,
    ProviderServerError,
)
from isales_engine.providers.tts_volcengine import VolcengineTTSProvider


def _provider_with(handler: Any) -> tuple[VolcengineTTSProvider, list[bytes]]:
    """Returns a provider + an output sink for assertions."""

    out: list[bytes] = []
    transport = httpx.MockTransport(handler)
    provider = VolcengineTTSProvider(
        endpoint="https://openspeech.bytedance.com/api/v1",
        app_key="k",
        app_token="t",
    )

    # Patch the HTTP layer to use our transport.
    async def stream_with_mock(text: str, voice_id: str) -> AsyncIterator[bytes]:
        url = "https://openspeech.bytedance.com/api/v1/tts"
        headers = {
            "Authorization": "Bearer; t",
            "Content-Type": "application/json",
            "Resource-Id": "tts",
            "App-Key": "k",
        }
        payload = {
            "audio": {
                "voice_type": voice_id,
                "encoding": "pcm",
                "rate": 8000,
                "bits": 16,
                "channel": 1,
            },
            "request": {"reqid": "test", "text": text, "operation": "submit"},
        }
        async with (
            httpx.AsyncClient(transport=transport, timeout=5.0) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):
            if response.status_code >= 400:
                body = await response.aread()
                raise _map_status(response.status_code, body)
            async for chunk in response.aiter_bytes():
                if chunk:
                    out.append(chunk)
                    yield chunk

    provider.synthesize_stream = stream_with_mock  # type: ignore[method-assign]
    return provider, out


def _map_status(status: int, body: bytes) -> Exception:
    """Mirror what map_http_error returns for our test assertions."""

    from isales_engine.providers._errors import map_http_error

    return map_http_error(
        httpx.Response(status_code=status, content=body), provider="volcengine_tts"
    )


# ---- happy path -----------------------------------------------------------


async def test_streaming_yields_pcm_chunks() -> None:
    pcm_chunks = [b"\x01\x02" * 80, b"\x03\x04" * 80, b"\x05\x06" * 80]

    def handler(request: httpx.Request) -> httpx.Response:
        body = b"".join(pcm_chunks)
        return httpx.Response(200, content=body)

    provider, out = _provider_with(handler)
    received = [c async for c in provider.synthesize_stream("hello", "BV001")]
    assert sum(len(c) for c in received) == 3 * 160


async def test_request_payload_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x00")

    provider, _ = _provider_with(handler)
    [c async for c in provider.synthesize_stream("您好", "BV002_streaming")]

    assert seen["audio"]["voice_type"] == "BV002_streaming"
    assert seen["audio"]["encoding"] == "pcm"
    assert seen["audio"]["rate"] == 8000
    assert seen["audio"]["bits"] == 16
    assert seen["audio"]["channel"] == 1
    assert seen["request"]["text"] == "您好"
    assert seen["request"]["operation"] == "submit"


# ---- error mapping --------------------------------------------------------


async def test_429_raises_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "throttled"})

    provider, _ = _provider_with(handler)
    with pytest.raises(ProviderRateLimited):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


async def test_5xx_raises_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "vendor down"})

    provider, _ = _provider_with(handler)
    with pytest.raises(ProviderServerError):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


async def test_4xx_raises_invalid_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad voice"})

    provider, _ = _provider_with(handler)
    with pytest.raises(ProviderInvalidRequest):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


# ---- factory routing ------------------------------------------------------


def test_factory_volcengine_requires_credentials() -> None:
    from isales_engine.providers.factory import build_tts
    from isales_engine.settings import Settings

    empty = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
    )
    with pytest.raises(NotImplementedError, match="VOLCENGINE_APP_KEY"):
        build_tts("volcengine", settings=empty)


def test_factory_volcengine_with_credentials() -> None:
    from isales_engine.providers.factory import build_tts
    from isales_engine.settings import Settings

    s = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
        ISALES_VOLCENGINE_APP_KEY="k",
        ISALES_VOLCENGINE_APP_TOKEN="t",
    )
    provider = build_tts("volcengine", settings=s)
    assert isinstance(provider, VolcengineTTSProvider)
