"""Smoke tests for the PR #1 skeleton."""

from __future__ import annotations

from isales_engine import __version__
from isales_engine.settings import Settings


def test_version_present() -> None:
    assert __version__ == "0.1.0"


def test_settings_load_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ISALES_DATABASE_URL", "postgresql+asyncpg://x/y")
    monkeypatch.setenv("ISALES_REDIS_URL", "redis://localhost:6379/0")

    settings = Settings()

    assert settings.database_url == "postgresql+asyncpg://x/y"
    assert settings.redis_url == "redis://localhost:6379/0"
    # Defaults are wired correctly.
    assert settings.engine_llm_provider == "mock"
    assert settings.engine_telephony_mode == "mock"
    assert settings.engine_pipeline_default_timeout_ms == 8000
    assert settings.engine_dial_queue == "engine:dial"
    assert settings.engine_concurrency_key == "isales:concurrency:active"


def test_settings_overrides_via_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ISALES_DATABASE_URL", "postgresql+asyncpg://x/y")
    monkeypatch.setenv("ISALES_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ISALES_ENGINE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("ISALES_ENGINE_PIPELINE_DEFAULT_TIMEOUT_MS", "12000")

    settings = Settings()

    assert settings.engine_llm_provider == "openai"
    assert settings.engine_pipeline_default_timeout_ms == 12000
