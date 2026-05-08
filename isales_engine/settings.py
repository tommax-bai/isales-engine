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

    # ---- impl-engine-providers (stage 5) --------------------------------

    # Token budget per call: emit a WARN log when exceeded. Hook for a
    # future EngineEvent.TokenBudgetExceeded notification (out of scope for
    # this change).
    engine_token_budget_per_call: int = Field(
        default=50_000, alias="ISALES_ENGINE_TOKEN_BUDGET_PER_CALL"
    )

    # Volcengine (火山引擎) shared credentials for ASR / TTS / LLM (豆包).
    volcengine_app_key: str | None = Field(
        default=None, alias="ISALES_VOLCENGINE_APP_KEY"
    )
    volcengine_app_token: str | None = Field(
        default=None, alias="ISALES_VOLCENGINE_APP_TOKEN"
    )
    volcengine_llm_model: str = Field(
        default="doubao-pro-32k", alias="ISALES_VOLCENGINE_LLM_MODEL"
    )
    volcengine_asr_endpoint: str = Field(
        default="wss://openspeech.bytedance.com/api/v3/asr",
        alias="ISALES_VOLCENGINE_ASR_ENDPOINT",
    )
    volcengine_tts_voice_id_default: str = Field(
        default="BV001_streaming", alias="ISALES_VOLCENGINE_TTS_VOICE_ID_DEFAULT"
    )

    # OpenAI (chat completions; supports OpenAI-compatible base URLs incl.
    # Azure OpenAI / 第三方兼容服务).
    openai_api_key: str | None = Field(default=None, alias="ISALES_OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="ISALES_OPENAI_BASE_URL"
    )
    openai_llm_model: str = Field(
        default="gpt-4o-mini", alias="ISALES_OPENAI_LLM_MODEL"
    )

    # Live-API integration tests opt-in. CI must NOT set this.
    live_provider_tests: bool = Field(
        default=False, alias="ISALES_LIVE_PROVIDER_TESTS"
    )


def load_settings() -> Settings:
    return Settings()
