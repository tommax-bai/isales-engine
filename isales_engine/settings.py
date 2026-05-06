"""Process-level settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISALES_", case_sensitive=False)

    database_url: str = Field(alias="ISALES_DATABASE_URL")
    redis_url: str = Field(alias="ISALES_REDIS_URL")

    engine_llm_provider: str = Field(default="mock", alias="ISALES_ENGINE_LLM_PROVIDER")
    engine_asr_provider: str = Field(default="mock", alias="ISALES_ENGINE_ASR_PROVIDER")
    engine_tts_provider: str = Field(default="mock", alias="ISALES_ENGINE_TTS_PROVIDER")
    engine_telephony_mode: str = Field(default="mock", alias="ISALES_ENGINE_TELEPHONY_MODE")

    engine_pipeline_default_timeout_ms: int = Field(
        default=8000, alias="ISALES_ENGINE_PIPELINE_DEFAULT_TIMEOUT_MS"
    )
    engine_max_no_progress_seconds: int = Field(
        default=60, alias="ISALES_ENGINE_MAX_NO_PROGRESS_SECONDS"
    )
    engine_mock_connect_delay_ms: int = Field(
        default=200, alias="ISALES_ENGINE_MOCK_CONNECT_DELAY_MS"
    )
    engine_graceful_shutdown_timeout_s: int = Field(
        default=30, alias="ISALES_ENGINE_GRACEFUL_SHUTDOWN_TIMEOUT_S"
    )

    engine_dial_queue: str = Field(default="engine:dial", alias="ISALES_ENGINE_DIAL_QUEUE")
    engine_dlq: str = Field(default="engine:dlq", alias="ISALES_ENGINE_DLQ")
    engine_call_ended_queue: str = Field(
        default="engine:worker:call-ended", alias="ISALES_ENGINE_CALL_ENDED_QUEUE"
    )
    engine_concurrency_key: str = Field(
        default="isales:concurrency:active", alias="ISALES_ENGINE_CONCURRENCY_KEY"
    )


def load_settings() -> Settings:
    return Settings()
