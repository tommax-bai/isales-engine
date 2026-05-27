"""Provider factory — routes provider name to a concrete implementation.

Spec: provider-abc § Requirement: Provider ABC 集中定义在 isales-common;
      provider-credential capability § "engine 启动期加载凭据，运行期不
      再读 DB"。

Stage-4 mock providers are always available (used by tests + dev mock mode).
Real providers (volcengine / openai / dashscope) read credentials from a
``CredentialStore`` (装载自 ``provider_credential`` 表) — env 不再持有
provider 密钥。

凭据来源:
- ``mock``: 不需凭据。
- ``volcengine``: ``app_key`` + ``app_token`` (双密钥)。
- ``openai`` / ``dashscope``: ``api_key`` + ``endpoint`` (可选)。

调用方 (engine main.py / 测试) 负责在 startup 装载 CredentialStore 并
透传给 ``build_*``；store=None 仅在 mock 路径下合法 (NotImplementedError
其他路径)。
"""

from __future__ import annotations

from isales_common.credentials import CredentialStore
from isales_common.providers.asr import ASRProvider
from isales_common.providers.llm import LLMProvider
from isales_common.providers.tts import TTSProvider

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.llm_openai_compatible import (
    OpenAICompatibleLLMProvider,
)
from isales_engine.providers.tts_mock import TextLengthMockTTS

# Provider names known to the factory. Adding a new vendor MUST update
# this constant + the corresponding build_* branch + the
# ``ALLOWED_PROVIDER_IDS`` in isales-api routers/provider_credentials.py.
KNOWN_LLM_PROVIDERS: frozenset[str] = frozenset(
    {"mock", "volcengine", "openai", "dashscope"}
)
KNOWN_ASR_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine"})
KNOWN_TTS_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine"})


# Provider-specific defaults for endpoint when the credential row is
# missing the optional `endpoint` field. Volcengine LLM uses ark.
_DEFAULT_ENDPOINT: dict[str, str] = {
    "volcengine_llm": "https://ark.cn-beijing.volces.com/api/v3",
    "openai_llm": "https://api.openai.com/v1",
    "dashscope_llm": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "volcengine_asr": "wss://openspeech.bytedance.com/api/v3/asr",
    "volcengine_tts": "https://openspeech.bytedance.com/api/v1",
}

_DEFAULT_MODEL: dict[str, str] = {
    "volcengine": "doubao-pro-32k",
    "openai": "gpt-4o-mini",
    "dashscope": "qwen-plus",
}


def _require(store: CredentialStore | None, provider: str) -> CredentialStore:
    """非 mock 路径要求 store 必须传入。"""
    if store is None:
        raise NotImplementedError(
            f"provider {provider!r} requires a CredentialStore; "
            "engine main.py must load via CredentialStore.from_db(session) "
            "at startup. See provider-credential spec § engine 启动期加载凭据."
        )
    return store


def _require_field(store: CredentialStore, provider: str, field: str) -> str:
    value = store.get(provider, field)
    if not value:
        raise NotImplementedError(
            f"provider {provider!r} requires field {field!r} in "
            f"provider_credential; configure via UI «模型厂商» or "
            f"`isales-cred-migrate import-env`"
        )
    return value


def build_llm(
    name: str,
    *,
    store: CredentialStore | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Return a LLM provider instance.

    ``model`` overrides the credential-configured default when provided
    (used for campaign-level model selection — role_config.model).
    """
    if name == "mock":
        return KeywordDrivenMockLLM()
    if name not in KNOWN_LLM_PROVIDERS:
        raise NotImplementedError(
            f"LLM provider {name!r} not supported "
            f"(known: {sorted(KNOWN_LLM_PROVIDERS)})"
        )

    s = _require(store, name)

    if name == "volcengine":
        api_key = _require_field(s, "volcengine", "app_token")
        return OpenAICompatibleLLMProvider(
            provider="volcengine",
            api_key=api_key,
            base_url=s.get("volcengine", "endpoint") or _DEFAULT_ENDPOINT["volcengine_llm"],
            model=model or s.get("volcengine", "default_model") or _DEFAULT_MODEL["volcengine"],
        )
    if name == "openai":
        api_key = _require_field(s, "openai", "api_key")
        return OpenAICompatibleLLMProvider(
            provider="openai",
            api_key=api_key,
            base_url=s.get("openai", "endpoint") or _DEFAULT_ENDPOINT["openai_llm"],
            model=model or s.get("openai", "default_model") or _DEFAULT_MODEL["openai"],
        )
    if name == "dashscope":
        api_key = _require_field(s, "dashscope", "api_key")
        # DashScope OpenAI-compatible mode (api 表面相同，base_url 不同)。
        return OpenAICompatibleLLMProvider(
            provider="dashscope",
            api_key=api_key,
            base_url=s.get("dashscope", "endpoint") or _DEFAULT_ENDPOINT["dashscope_llm"],
            model=model or s.get("dashscope", "default_model") or _DEFAULT_MODEL["dashscope"],
        )
    raise NotImplementedError(  # pragma: no cover  - exhaustive above
        f"LLM provider {name!r} not supported"
    )


def build_asr(
    name: str, *, store: CredentialStore | None = None
) -> ASRProvider:
    if name == "mock":
        return ScriptedMockASR()
    if name not in KNOWN_ASR_PROVIDERS:
        raise NotImplementedError(
            f"ASR provider {name!r} not supported "
            f"(known: {sorted(KNOWN_ASR_PROVIDERS)})"
        )
    s = _require(store, name)
    if name == "volcengine":
        # 豆包 V3 SAUC ASR 两套鉴权 (vendor 文档 § 鉴权), 跟 TTS V3 同一模型:
        #   1. 新版控制台: X-Api-Key 单 UUID (provider_credential.volcengine.api_key)
        #   2. 旧版控制台: X-Api-App-Key + X-Api-Access-Key (复用 volcengine.app_key
        #      / app_token 的现存值, 注意 ASR 用 X-Api-App-Key header 名,TTS 用
        #      X-Api-App-Id, vendor 文档命名不一致但本质同一字段)
        # resource_id 决定 ASR 模型版本 (volc.bigasr.sauc.duration 1.0 /
        # volc.seedasr.sauc.duration 2.0 / .concurrent 并发版).
        from isales_engine.providers.asr_volcengine import (  # noqa: PLC0415
            DEFAULT_ASR_RESOURCE_ID,
            VolcengineASRProvider,
        )

        api_key = s.get("volcengine", "api_key")
        resource_id = (
            s.get("volcengine", "asr_resource_id") or DEFAULT_ASR_RESOURCE_ID
        )
        if api_key:
            return VolcengineASRProvider(
                api_key=api_key,
                resource_id=resource_id,
            )
        # legacy fallback
        app_key = _require_field(s, "volcengine", "app_key")
        app_token = _require_field(s, "volcengine", "app_token")
        return VolcengineASRProvider(
            app_key=app_key,
            access_key=app_token,
            resource_id=resource_id,
        )
    raise NotImplementedError(name)  # pragma: no cover


def build_tts(
    name: str, *, store: CredentialStore | None = None
) -> TTSProvider:
    if name == "mock":
        return TextLengthMockTTS()
    if name not in KNOWN_TTS_PROVIDERS:
        raise NotImplementedError(
            f"TTS provider {name!r} not supported "
            f"(known: {sorted(KNOWN_TTS_PROVIDERS)})"
        )
    s = _require(store, name)
    if name == "volcengine":
        # 豆包 V3 SSE TTS 支持两套鉴权 (vendor 文档 § 2.1):
        #   1. 新版控制台 (推荐): X-Api-Key 单 UUID
        #      → provider_credential.volcengine.api_key
        #   2. 旧版控制台 (fallback): X-Api-App-Id + X-Api-Access-Key
        #      → provider_credential.volcengine.{app_key, app_token}
        # 新版优先, 没配再降到旧版三件套. resource_id 决定模型版本 (seed-tts-2.0
        # / seed-tts-1.0 / seed-icl-2.0 etc.), 不传走默认 seed-tts-2.0.
        from isales_engine.providers.tts_volcengine import VolcengineTTSProvider

        api_key = s.get("volcengine", "api_key")
        resource_id = s.get("volcengine", "tts_resource_id") or "seed-tts-2.0"
        if api_key:
            return VolcengineTTSProvider(
                api_key=api_key,
                resource_id=resource_id,
            )
        # legacy 3-tuple fallback
        app_key = _require_field(s, "volcengine", "app_key")
        app_token = _require_field(s, "volcengine", "app_token")
        return VolcengineTTSProvider(
            app_id=app_key,
            access_key=app_token,
            resource_id=resource_id,
        )
    raise NotImplementedError(name)  # pragma: no cover
