"""Provider factory.

Spec: provider-abc § Requirement: Provider ABC 集中定义在 isales-common.

PR #4 wires the mock variants only; real providers (OpenAI / Volcengine /
Alibaba) are added by stage 5's ``impl-engine-providers`` change.
"""

from __future__ import annotations

from isales_common.providers.asr import ASRProvider
from isales_common.providers.llm import LLMProvider
from isales_common.providers.tts import TTSProvider

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.tts_mock import TextLengthMockTTS


def build_llm(name: str) -> LLMProvider:
    if name == "mock":
        return KeywordDrivenMockLLM()
    raise NotImplementedError(
        f"LLM provider {name!r} not wired — stage 5 (impl-engine-providers)"
    )


def build_asr(name: str) -> ASRProvider:
    if name == "mock":
        return ScriptedMockASR()
    raise NotImplementedError(
        f"ASR provider {name!r} not wired — stage 5 (impl-engine-providers)"
    )


def build_tts(name: str) -> TTSProvider:
    if name == "mock":
        return TextLengthMockTTS()
    raise NotImplementedError(
        f"TTS provider {name!r} not wired — stage 5 (impl-engine-providers)"
    )
