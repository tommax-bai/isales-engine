"""Provider factory — routes provider name to a concrete implementation.

Spec: provider-abc § Requirement: Provider ABC 集中定义在 isales-common;
      impl-engine-providers proposal § "provider 工厂增量".

Stage-4 mock providers are always available (used by tests + the default
stage-4 deployment). Real providers (volcengine / openai) are wired by
PR #2-#5 of impl-engine-providers; until then they raise
``NotImplementedError`` with a clear pointer to the change.
"""

from __future__ import annotations

from isales_common.providers.asr import ASRProvider
from isales_common.providers.llm import LLMProvider
from isales_common.providers.tts import TTSProvider

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.llm_openai_compatible import (
    OpenAICompatibleLLMProvider,
)
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.settings import Settings

# Provider names known to the factory. Adding a new vendor MUST update this
# constant + the corresponding build_* branch + the impl-engine-providers
# (or a successor) change spec.
KNOWN_LLM_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine", "openai"})
KNOWN_ASR_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine"})
KNOWN_TTS_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine"})


def build_llm(
    name: str,
    *,
    model: str | None = None,
    settings: Settings | None = None,
) -> LLMProvider:
    """Return a LLM provider instance.

    ``model`` overrides the env-configured default model when provided
    (used for Campaign-level model selection — PR #8). When ``settings``
    is omitted the factory loads it once from env; tests can pass an
    explicit ``Settings`` to avoid env contamination.
    """

    if name == "mock":
        return KeywordDrivenMockLLM()
    if name not in KNOWN_LLM_PROVIDERS:
        raise NotImplementedError(
            f"LLM provider {name!r} not supported (known: {sorted(KNOWN_LLM_PROVIDERS)})"
        )
    if settings is None:
        settings = Settings()
    if name == "volcengine":
        if not settings.volcengine_app_token:
            raise NotImplementedError(
                "LLM provider 'volcengine' requires ISALES_VOLCENGINE_APP_TOKEN"
            )
        return OpenAICompatibleLLMProvider(
            provider="volcengine",
            api_key=settings.volcengine_app_token,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model=model or settings.volcengine_llm_model,
        )
    if name == "openai":
        if not settings.openai_api_key:
            raise NotImplementedError(
                "LLM provider 'openai' requires ISALES_OPENAI_API_KEY"
            )
        return OpenAICompatibleLLMProvider(
            provider="openai",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=model or settings.openai_llm_model,
        )
    raise NotImplementedError(
        f"LLM provider {name!r} not supported (known: {sorted(KNOWN_LLM_PROVIDERS)})"
    )


def build_asr(name: str, *, settings: Settings | None = None) -> ASRProvider:
    if name == "mock":
        return ScriptedMockASR()
    if name not in KNOWN_ASR_PROVIDERS:
        raise NotImplementedError(
            f"ASR provider {name!r} not supported (known: {sorted(KNOWN_ASR_PROVIDERS)})"
        )
    if settings is None:
        settings = Settings()
    if name == "volcengine":
        if not (settings.volcengine_app_key and settings.volcengine_app_token):
            raise NotImplementedError(
                "ASR provider 'volcengine' requires "
                "ISALES_VOLCENGINE_APP_KEY + ISALES_VOLCENGINE_APP_TOKEN"
            )
        from isales_engine.providers.asr_volcengine import VolcengineASRProvider

        return VolcengineASRProvider(
            endpoint=settings.volcengine_asr_endpoint,
            app_key=settings.volcengine_app_key,
            app_token=settings.volcengine_app_token,
        )
    raise NotImplementedError(name)


def build_tts(name: str, *, settings: Settings | None = None) -> TTSProvider:
    if name == "mock":
        return TextLengthMockTTS()
    if name not in KNOWN_TTS_PROVIDERS:
        raise NotImplementedError(
            f"TTS provider {name!r} not supported (known: {sorted(KNOWN_TTS_PROVIDERS)})"
        )
    if settings is None:
        settings = Settings()
    if name == "volcengine":
        if not (settings.volcengine_app_key and settings.volcengine_app_token):
            raise NotImplementedError(
                "TTS provider 'volcengine' requires "
                "ISALES_VOLCENGINE_APP_KEY + ISALES_VOLCENGINE_APP_TOKEN"
            )
        # Volcengine streaming TTS endpoint (HTTP). Override via env if the
        # account is on a different region.
        from isales_engine.providers.tts_volcengine import VolcengineTTSProvider

        endpoint = "https://openspeech.bytedance.com/api/v1"
        return VolcengineTTSProvider(
            endpoint=endpoint,
            app_key=settings.volcengine_app_key,
            app_token=settings.volcengine_app_token,
        )
    raise NotImplementedError(name)
