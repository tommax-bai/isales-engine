"""Process-level settings loaded from environment variables.

provider 凭据 (volcengine app_key/app_token, openai api_key 等) 不再
从 env 读取，转走 provider_credential 表 + CredentialStore。本 Settings
对象只持有非凭据配置 (DB URL、provider 名字、telephony 模式等)。

凭据 SSOT 切换历史: impl-provider-credential-db-ssot (2026-05-24)。
"""

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
    # mock | real | rtc — rtc = cloud-edge gRPC + Aliyun RTC (arch-cloud-edge-split A2).
    engine_telephony_mode: str = Field(default="mock", alias="ISALES_ENGINE_TELEPHONY_MODE")
    engine_telephony_socket_path: str = Field(
        default="/var/run/isales/modem.sock", alias="ISALES_ENGINE_TELEPHONY_SOCKET_PATH"
    )
    engine_telephony_dial_timeout_s: float = Field(
        default=60.0, alias="ISALES_ENGINE_TELEPHONY_DIAL_TIMEOUT_S"
    )

    # ---- arch-cloud-edge-split (mode="rtc") -----------------------------

    engine_cloud_edge_grpc_bind: str = Field(
        default="0.0.0.0:50051", alias="ISALES_ENGINE_CLOUD_EDGE_GRPC_BIND"
    )
    # A2 assumes a single edge_device_id per engine process (single-tenant);
    # multi-edge → engine routing arrives with C2.
    # Non-empty → legacy single-Edge mode (all dials routed to this edge).
    # Empty string (default) → dynamic routing mode (device → edge mapping
    # is built from Heartbeat.DeviceHealth frames at runtime).
    engine_edge_device_id: str = Field(
        default="", alias="ISALES_ENGINE_EDGE_DEVICE_ID"
    )
    engine_rtc_app_id: str = Field(default="", alias="ISALES_RTC_APP_ID")
    # DingRTC AppKey — NOT a model provider credential, NOT under
    # provider_credential SSOT. Stays in env (engine-only, never crosses
    # cloud-edge). See impl-provider-credential-db-ssot design § Non-Goals.
    engine_rtc_app_key: str = Field(default="", alias="ISALES_RTC_APP_KEY")
    # vendor → real DingRTC SDK (loads dingrtc_pywrap extension, requires
    # vendor SDK on $ISALES_DINGRTC_LINUX_SDK_PATH and LD_LIBRARY_PATH);
    # in_memory → InMemoryDingRtcChannel test double (local dev / smoke).
    engine_rtc_sdk_kind: str = Field(
        default="vendor", alias="ISALES_ENGINE_RTC_SDK_KIND"
    )
    # Shared with isales-api edge-token mint (same secret, audience=cloud-edge).
    engine_edge_token_jwt_secret: str = Field(
        default="", alias="ISALES_JWT_SECRET"
    )

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
    engine_grpc_send_timeout_s: float = Field(
        default=5.0, alias="ISALES_ENGINE_GRPC_SEND_TIMEOUT_S"
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

    # 非凭据 — 默认音色 ID (volcengine TTS-specific 配置，不属密钥)；
    # 留 env 兼容 sample 化部署 (后续 voice_id 可考虑也移到 DB)。
    volcengine_tts_voice_id_default: str = Field(
        default="BV001_streaming", alias="ISALES_VOLCENGINE_TTS_VOICE_ID_DEFAULT"
    )

    # ---- impl-provider-credential-db-ssot (2026-05-24) ------------------

    # True (default) — engine startup MUST 从 DB 装载 provider 凭据；装载
    # 失败 (DB 不可达 / Fernet 解密失败 / 表空) → 进程退出。
    # False — 仅 dev/CI 用；装载失败时强制走 mock provider。
    credentials_required: bool = Field(
        default=True, alias="ISALES_CREDENTIALS_REQUIRED"
    )

    # Live-API integration tests opt-in. CI must NOT set this.
    live_provider_tests: bool = Field(
        default=False, alias="ISALES_LIVE_PROVIDER_TESTS"
    )


def load_settings() -> Settings:
    return Settings()
