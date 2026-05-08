"""Tests for providers/_errors.py + factory routing."""

from __future__ import annotations

import httpx
import pytest

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
    from isales_engine.settings import Settings

    empty = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
    )
    with pytest.raises(NotImplementedError, match="VOLCENGINE_APP_TOKEN"):
        build_llm("volcengine", settings=empty)
    with pytest.raises(NotImplementedError, match="OPENAI_API_KEY"):
        build_llm("openai", settings=empty)


def test_build_llm_volcengine_with_credentials() -> None:
    from isales_engine.providers.llm_openai_compatible import (
        OpenAICompatibleLLMProvider,
    )
    from isales_engine.settings import Settings

    s = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
        ISALES_VOLCENGINE_APP_TOKEN="t",
    )
    provider = build_llm("volcengine", settings=s)
    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_build_llm_openai_with_credentials() -> None:
    from isales_engine.providers.llm_openai_compatible import (
        OpenAICompatibleLLMProvider,
    )
    from isales_engine.settings import Settings

    s = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
        ISALES_OPENAI_API_KEY="sk-x",
    )
    provider = build_llm("openai", settings=s)
    assert isinstance(provider, OpenAICompatibleLLMProvider)


def test_build_llm_with_explicit_model_override() -> None:
    """PR #8 use-case: Campaign-level model selection."""

    from isales_engine.settings import Settings

    s = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
        ISALES_OPENAI_API_KEY="sk-x",
    )
    provider = build_llm("openai", model="gpt-4o", settings=s)
    assert provider._model == "gpt-4o"  # type: ignore[attr-defined]


def test_build_asr_mock_works() -> None:
    assert build_asr("mock") is not None


def test_build_asr_volcengine_signals_pending_pr() -> None:
    with pytest.raises(NotImplementedError, match="PR #4"):
        build_asr("volcengine")


def test_build_tts_volcengine_signals_pending_pr() -> None:
    with pytest.raises(NotImplementedError, match="PR #5"):
        build_tts("volcengine")


def test_unknown_provider_lists_known_set() -> None:
    with pytest.raises(NotImplementedError, match="known:"):
        build_llm("anthropic")


# ---- Settings -------------------------------------------------------------


def test_settings_loads_new_env_vars(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ISALES_DATABASE_URL", "postgresql+asyncpg://x/y")
    monkeypatch.setenv("ISALES_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ISALES_VOLCENGINE_APP_KEY", "k")
    monkeypatch.setenv("ISALES_VOLCENGINE_APP_TOKEN", "t")
    monkeypatch.setenv("ISALES_OPENAI_API_KEY", "sk-x")

    from isales_engine.settings import Settings

    s = Settings()
    assert s.volcengine_app_key == "k"
    assert s.volcengine_app_token == "t"
    assert s.volcengine_llm_model == "doubao-pro-32k"
    assert s.openai_api_key == "sk-x"
    assert s.openai_base_url == "https://api.openai.com/v1"
    assert s.openai_llm_model == "gpt-4o-mini"
    assert s.engine_token_budget_per_call == 50_000
    assert s.live_provider_tests is False
