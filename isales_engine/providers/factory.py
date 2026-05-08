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
from isales_engine.providers.tts_mock import TextLengthMockTTS

# Provider names known to the factory. Adding a new vendor MUST update this
# constant + the corresponding build_* branch + the impl-engine-providers
# (or a successor) change spec.
KNOWN_LLM_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine", "openai"})
KNOWN_ASR_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine"})
KNOWN_TTS_PROVIDERS: frozenset[str] = frozenset({"mock", "volcengine"})


def build_llm(name: str, *, model: str | None = None) -> LLMProvider:
    """Return a LLM provider instance.

    ``model`` is currently advisory — real implementations will use it to
    select the vendor-specific endpoint / payload (PR #8 of
    impl-engine-providers). The mock provider ignores it.
    """

    if name == "mock":
        return KeywordDrivenMockLLM()
    if name == "volcengine":
        raise NotImplementedError(
            "LLM provider 'volcengine' not yet wired — see "
            "impl-engine-providers PR #2"
        )
    if name == "openai":
        raise NotImplementedError(
            "LLM provider 'openai' not yet wired — see impl-engine-providers PR #3"
        )
    raise NotImplementedError(
        f"LLM provider {name!r} not supported (known: {sorted(KNOWN_LLM_PROVIDERS)})"
    )


def build_asr(name: str) -> ASRProvider:
    if name == "mock":
        return ScriptedMockASR()
    if name == "volcengine":
        raise NotImplementedError(
            "ASR provider 'volcengine' not yet wired — see "
            "impl-engine-providers PR #4"
        )
    raise NotImplementedError(
        f"ASR provider {name!r} not supported (known: {sorted(KNOWN_ASR_PROVIDERS)})"
    )


def build_tts(name: str) -> TTSProvider:
    if name == "mock":
        return TextLengthMockTTS()
    if name == "volcengine":
        raise NotImplementedError(
            "TTS provider 'volcengine' not yet wired — see "
            "impl-engine-providers PR #5"
        )
    raise NotImplementedError(
        f"TTS provider {name!r} not supported (known: {sorted(KNOWN_TTS_PROVIDERS)})"
    )
