"""Tests for VolcengineTTSProvider over the V3 SSE protocol.

Uses ``httpx.MockTransport`` injected into the provider's persistent client
to verify SSE framing, ProviderError mapping, and the pipeline-latency-tail
§ C connection-reuse + ``aclose()`` behavior. Real-vendor smoke tests live
in tests/test_real_providers.py and require ``ISALES_LIVE_PROVIDER_TESTS=1``
(not run in CI).

These tests replaced the legacy V1 (``/api/v1/tts`` with ``endpoint`` /
``app_key`` / ``app_token`` kwargs) suite, which monkeypatched
``synthesize_stream`` wholesale and so never exercised the real provider
after the V3 SSE rewrite (the constructor no longer accepts those kwargs).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from isales_engine.providers._errors import (
    ProviderInvalidRequest,
    ProviderRateLimited,
    ProviderServerError,
)
from isales_engine.providers.tts_volcengine import VolcengineTTSProvider


def _sse(*frames: tuple[str, dict[str, Any]]) -> bytes:
    """Build a V3 SSE byte body from (event_id, data_obj) pairs."""
    parts: list[str] = []
    for event_id, data_obj in frames:
        parts.append(f"event: {event_id}")
        parts.append(f"data: {json.dumps(data_obj)}")
        parts.append("")  # blank line = frame terminator
    return ("\n".join(parts) + "\n").encode("utf-8")


def _audio_frame(pcm: bytes) -> tuple[str, dict[str, Any]]:
    return ("352", {"code": 0, "message": "", "data": base64.b64encode(pcm).decode()})


_FINISH = ("152", {"code": 20000000, "message": "OK", "data": None})


def _provider_with(handler: Any) -> VolcengineTTSProvider:
    """Construct a provider, then swap its persistent client for one backed
    by the given MockTransport — mirrors how the real provider reuses a
    single ``self._client`` across sentences."""
    provider = VolcengineTTSProvider(api_key="key-uuid")
    transport = httpx.MockTransport(handler)
    provider._client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return provider


# ---- happy path -----------------------------------------------------------


async def test_streaming_yields_pcm_chunks() -> None:
    pcm_chunks = [b"\x01\x02" * 80, b"\x03\x04" * 80, b"\x05\x06" * 80]

    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse(*[_audio_frame(c) for c in pcm_chunks], _FINISH)
        return httpx.Response(200, content=body)

    provider = _provider_with(handler)
    received = [c async for c in provider.synthesize_stream("hello", "BV001")]
    assert b"".join(received) == b"".join(pcm_chunks)


async def test_request_payload_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_sse(_audio_frame(b"\x00\x00"), _FINISH))

    provider = _provider_with(handler)
    [c async for c in provider.synthesize_stream("您好", "zh_female_test_uranus_bigtts")]

    assert seen["req_params"]["text"] == "您好"
    assert seen["req_params"]["speaker"] == "zh_female_test_uranus_bigtts"
    assert seen["req_params"]["audio_params"]["format"] == "pcm"


# ---- error mapping --------------------------------------------------------


async def test_429_raises_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "throttled"})

    provider = _provider_with(handler)
    with pytest.raises(ProviderRateLimited):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


async def test_5xx_raises_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "vendor down"})

    provider = _provider_with(handler)
    with pytest.raises(ProviderServerError):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


async def test_4xx_raises_invalid_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad voice"})

    provider = _provider_with(handler)
    with pytest.raises(ProviderInvalidRequest):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


async def test_session_failed_event_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse(("153", {"code": 40000001, "message": "bad speaker", "data": None}))
        return httpx.Response(200, content=body)

    provider = _provider_with(handler)
    with pytest.raises(ProviderInvalidRequest):
        async for _ in provider.synthesize_stream("x", "v"):
            pass


# ---- connection reuse (pipeline-latency-tail § C) -------------------------


async def test_reuses_same_client_across_sentences() -> None:
    """Consecutive synthesize_stream calls MUST go through the same
    persistent client (no per-sentence client rebuild)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=_sse(_audio_frame(b"\x10\x11"), _FINISH))

    provider = _provider_with(handler)
    client_before = provider._client
    for sentence in ("第一句", "第二句", "第三句"):
        async for _ in provider.synthesize_stream(sentence, "v"):
            pass

    assert calls["n"] == 3
    # Same client object reused for every sentence — not rebuilt per call.
    assert provider._client is client_before
    assert provider._client.is_closed is False


async def test_aclose_releases_client() -> None:
    provider = VolcengineTTSProvider(api_key="key-uuid")
    assert provider._client.is_closed is False
    await provider.aclose()
    assert provider._client.is_closed is True


# ---- factory routing ------------------------------------------------------


def test_factory_volcengine_requires_credentials() -> None:
    """缺凭据字段 = NotImplementedError (provider-credential SSOT)."""
    from isales_common.credentials import CredentialStore

    from isales_engine.providers.factory import build_tts

    empty_store = CredentialStore()
    with pytest.raises(NotImplementedError, match="app_key"):
        build_tts("volcengine", store=empty_store)


def test_factory_volcengine_with_credentials() -> None:
    from isales_common.credentials import CredentialStore

    from isales_engine.providers.factory import build_tts

    store = CredentialStore({"volcengine": {"app_key": "k", "app_token": "t"}})
    provider = build_tts("volcengine", store=store)
    assert isinstance(provider, VolcengineTTSProvider)
