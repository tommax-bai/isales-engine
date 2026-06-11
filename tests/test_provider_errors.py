"""Tests for providers/_errors.py + factory routing.

impl-provider-credential-db-ssot (2026-05-24) 把 provider 凭据从 env 切
到 DB-backed CredentialStore；factory.build_* 不再接 Settings，改接
CredentialStore。本测试同步更新。
"""

from __future__ import annotations

import httpx
import pytest
from isales_common.credentials import CredentialStore

from isales_engine.providers._errors import (
    ProviderInvalidRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
    map_http_error,
    map_transport_error,
)
from isales_engine.providers.factory import build_asr, build_llm, build_tts
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM

# ---- HTTP status → ProviderError mapping ----------------------------------


def _resp(status: int, body: str = "{}", headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=body.encode(),
        headers=headers or {"content-type": "application/json"},
    )


def test_http_429_maps_to_rate_limited() -> None:
    err = map_http_error(_resp(429, '{"code": "RPM_EXCEEDED"}'), provider="openai")
    assert isinstance(err, ProviderRateLimited)
    assert err.provider == "openai"
    assert err.vendor_code == "RPM_EXCEEDED"


def test_http_429_with_retry_after_header() -> None:
    err = map_http_error(
        _resp(429, "{}", headers={"Retry-After": "30", "content-type": "application/json"}),
        provider="openai",
    )
    assert isinstance(err, ProviderRateLimited)
    assert err.retry_after_seconds == 30.0


def test_http_5xx_maps_to_server_error() -> None:
    for status in (500, 502, 503, 504):
        err = map_http_error(_resp(status), provider="volcengine")
        assert isinstance(err, ProviderServerError)


def test_http_4xx_maps_to_invalid_request() -> None:
    for status in (400, 401, 403, 404, 422):
        err = map_http_error(_resp(status), provider="openai")
        assert isinstance(err, ProviderInvalidRequest)


def test_http_2xx_treated_as_caller_misuse() -> None:
    # Documented contract: caller only invokes on errors. If misused on 2xx,
    # we still return a ProviderError instead of None.
    err = map_http_error(_resp(200), provider="openai")
    assert isinstance(err, ProviderInvalidRequest)


def test_vendor_code_extraction_from_nested_error_object() -> None:
    body = '{"error": {"code": "invalid_api_key", "message": "no good"}}'
    err = map_http_error(_resp(401, body), provider="openai")
    assert err.vendor_code == "invalid_api_key"


# ---- Transport errors -----------------------------------------------------


def test_transport_timeout_maps_to_provider_timeout() -> None:
    exc = httpx.ReadTimeout("read timed out")
    err = map_transport_error(exc, provider="openai")
    assert isinstance(err, ProviderTimeout)


def test_transport_connect_error_maps_to_server_error() -> None:
    exc = httpx.ConnectError("dns failure")
    err = map_transport_error(exc, provider="openai")
    assert isinstance(err, ProviderServerError)


def test_transport_remote_protocol_error_maps_to_server_error() -> None:
    exc = httpx.RemoteProtocolError("vendor closed early")
    err = map_transport_error(exc, provider="volcengine")
    assert isinstance(err, ProviderServerError)


# ---- Factory routing ------------------------------------------------------


def test_build_llm_mock_works() -> None:
    assert isinstance(build_llm("mock"), KeywordDrivenMockLLM)


def test_build_llm_real_providers_require_credentials() -> None:
    """store=None 或 store 缺字段 = NotImplementedError."""
    empty_store = CredentialStore()
    with pytest.raises(NotImplementedError, match="app_token"):
        build_llm("volcengine", store=empty_store)
    with pytest.raises(NotImplementedError, match="api_key"):
        build_llm("dashscope", store=empty_store)


def test_build_llm_volcengine_with_credentials() -> None:
    from isales_engine.providers.llm_openai_compatible import (
        OpenAICompatibleLLMProvider,
    )

    store = CredentialStore(
        {"volcengine": {"app_token": "t", "app_key": "k"}}
    )
    provider = build_llm("volcengine", store=store)
    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_build_llm_dashscope_with_credentials() -> None:
    from isales_engine.providers.llm_openai_compatible import (
        OpenAICompatibleLLMProvider,
    )

    store = CredentialStore({"dashscope": {"api_key": "sk-x"}})
    provider = build_llm("dashscope", store=store)
    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_build_llm_with_explicit_model_override() -> None:
    """Campaign-level model selection: 显式 model 参数覆盖 store default。"""
    store = CredentialStore(
        {"dashscope": {"api_key": "sk-x", "default_model": "qwen-plus"}}
    )
    provider = build_llm("dashscope", store=store, model="qwen-max")
    assert provider._model == "qwen-max"  # type: ignore[attr-defined]


def test_build_asr_mock_works() -> None:
    assert build_asr("mock") is not None


def test_build_asr_volcengine_requires_credentials() -> None:
    """缺 app_key / app_token 任一字段 = NotImplementedError."""
    empty_store = CredentialStore()
    with pytest.raises(NotImplementedError, match="app_key"):
        build_asr("volcengine", store=empty_store)


def test_build_asr_volcengine_with_credentials() -> None:
    from isales_engine.providers.asr_volcengine import VolcengineASRProvider

    store = CredentialStore(
        {"volcengine": {"app_key": "k", "app_token": "t"}}
    )
    provider = build_asr("volcengine", store=store)
    assert isinstance(provider, VolcengineASRProvider)


def test_build_tts_volcengine_requires_credentials() -> None:
    # TTS 构造下沉 isales-common 后, 缺凭据由 build_volcengine_tts 抛
    # ProviderInvalidRequest (campaign-greeting-tts-preview § 决策 1), 不再是
    # NotImplementedError — LLM / ASR 仍走 engine _require_field 的 NotImplementedError.
    from isales_common.providers._errors import ProviderInvalidRequest

    empty_store = CredentialStore()
    with pytest.raises(ProviderInvalidRequest, match="app_key"):
        build_tts("volcengine", store=empty_store)


def test_unknown_provider_lists_known_set() -> None:
    with pytest.raises(NotImplementedError, match="known:"):
        build_llm("anthropic")


# ---- Settings -------------------------------------------------------------


def test_settings_no_longer_carries_provider_secrets(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """SSOT 切换后 Settings 不再持有 provider 密钥；env 变量被读取但不
    expose 在 Settings 属性上 (pydantic-settings 默认 ignore extra)。"""
    monkeypatch.setenv("ISALES_DATABASE_URL", "postgresql+asyncpg://x/y")
    monkeypatch.setenv("ISALES_REDIS_URL", "redis://localhost:6379/0")
    # 老 env 字段即使设置也不该 expose 在 Settings 上
    monkeypatch.setenv("ISALES_VOLCENGINE_APP_KEY", "should-be-ignored")
    monkeypatch.setenv("ISALES_OPENAI_API_KEY", "should-also-be-ignored")

    from isales_engine.settings import Settings

    s = Settings()
    assert not hasattr(s, "volcengine_app_key")
    assert not hasattr(s, "volcengine_app_token")
    assert not hasattr(s, "openai_api_key")
    assert not hasattr(s, "openai_base_url")
    # 保留的非密字段
    assert s.engine_token_budget_per_call == 50_000
    assert s.live_provider_tests is False
    assert s.credentials_required is True  # 新加字段，默认 True
    assert s.volcengine_tts_voice_id_default == "BV001_streaming"


# ---- chat_stream error mapping (pipeline-stream-and-referee) ---------------
#
# chat_stream reuses map_http_error / map_transport_error, so the same vendor
# failures surface as the same ProviderError subclasses as chat(). One case
# per family here; the streaming-specific paths live in
# tests/test_provider_chat_stream.py.

from isales_engine.providers.llm_openai_compatible import (  # noqa: E402
    OpenAICompatibleLLMProvider,
)


def _stream_provider(handler) -> OpenAICompatibleLLMProvider:  # type: ignore[no-untyped-def]
    return OpenAICompatibleLLMProvider(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        transport=httpx.MockTransport(handler),
    )


async def _drain(provider: OpenAICompatibleLLMProvider) -> None:
    from isales_common.providers._models import Message

    async for _ in provider.chat_stream([Message(role="user", content="hi")]):
        pass


@pytest.mark.asyncio
async def test_chat_stream_500_maps_to_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"code": "OVERLOADED"})

    with pytest.raises(ProviderServerError):
        await _drain(_stream_provider(handler))


@pytest.mark.asyncio
async def test_chat_stream_4xx_maps_to_invalid_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "BAD"})

    with pytest.raises(ProviderInvalidRequest):
        await _drain(_stream_provider(handler))


@pytest.mark.asyncio
async def test_chat_stream_transport_timeout_maps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeout):
        await _drain(_stream_provider(handler))
