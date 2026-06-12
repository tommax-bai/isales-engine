"""Tests for provider factory + keyword-driven mocks."""

from __future__ import annotations

import asyncio
import json

import pytest
from isales_common.providers._models import Message

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.factory import build_asr, build_llm, build_tts
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.tts_mock import TextLengthMockTTS

# ---- factory ---------------------------------------------------------------


def test_factory_returns_mock_implementations() -> None:
    assert isinstance(build_llm("mock"), KeywordDrivenMockLLM)
    assert isinstance(build_asr("mock"), ScriptedMockASR)
    assert isinstance(build_tts("mock"), TextLengthMockTTS)


def test_factory_rejects_real_providers_without_credentials() -> None:
    """Non-mock providers raise NotImplementedError when:
    - store is None (caller didn't load credentials);
    - or the required field is missing from the store.

    impl-provider-credential-db-ssot 切换 SSOT 后，settings 不再持密钥；
    凭据走 CredentialStore (DB-backed)。
    """
    from isales_common.credentials import CredentialStore

    # 1) store=None → 任何非 mock provider 都报错。
    with pytest.raises(NotImplementedError):
        build_llm("dashscope")
    with pytest.raises(NotImplementedError):
        build_asr("volcengine_speech")
    with pytest.raises(NotImplementedError):
        build_tts("alibaba")

    # 2) store 在但缺字段 → 同样报错。
    empty_store = CredentialStore()
    with pytest.raises(NotImplementedError):
        build_llm("dashscope", store=empty_store)
    with pytest.raises(NotImplementedError):
        build_asr("volcengine_speech", store=empty_store)


def test_volcengine_asr_tts_no_longer_known() -> None:
    """split-model-and-speech-provider-config: 火山语音 id 改 volcengine_speech;
    旧的 ``volcengine`` 不再是 ASR/TTS provider (它现在只是 LLM id)。"""
    from isales_engine.providers.factory import (
        KNOWN_ASR_PROVIDERS,
        KNOWN_LLM_PROVIDERS,
        KNOWN_TTS_PROVIDERS,
    )

    assert "volcengine_speech" in KNOWN_ASR_PROVIDERS
    assert "volcengine_speech" in KNOWN_TTS_PROVIDERS
    assert "volcengine" not in KNOWN_ASR_PROVIDERS
    assert "volcengine" not in KNOWN_TTS_PROVIDERS
    # LLM 侧 volcengine 保留 (火山方舟 Ark)。
    assert "volcengine" in KNOWN_LLM_PROVIDERS
    assert "volcengine_speech" not in KNOWN_LLM_PROVIDERS


def test_build_llm_volcengine_uses_ark_api_key_not_app_token() -> None:
    """修 bug: build_llm('volcengine') 读 volcengine.api_key (ark key), 绝不读
    app_token (语音旧版 token) —— 后者已搬到 volcengine_speech。"""
    from isales_common.credentials import CredentialStore

    from isales_engine.providers.llm_openai_compatible import (
        OpenAICompatibleLLMProvider,
    )

    store = CredentialStore(
        {
            "volcengine": {"api_key": "ark-uuid-key", "default_model": "ep-xxx"},
            # 语音 token 放在 volcengine_speech, build_llm 绝不该读到它。
            "volcengine_speech": {"app_token": "speech-token-should-not-leak"},
        }
    )
    provider = build_llm("volcengine", store=store)
    assert isinstance(provider, OpenAICompatibleLLMProvider)
    assert provider._api_key == "ark-uuid-key"

    # 缺 ark api_key → 报错 (即便 volcengine_speech.app_token 存在也不兜底)。
    no_ark = CredentialStore(
        {"volcengine_speech": {"app_token": "speech-token"}}
    )
    with pytest.raises(NotImplementedError):
        build_llm("volcengine", store=no_ark)


def test_build_tts_reads_volcengine_speech_new_console() -> None:
    """build_tts('volcengine_speech') 走 volcengine_speech 的 X-Api-Key。"""
    from isales_common.credentials import CredentialStore

    from isales_common.providers.tts_volcengine import VolcengineTTSProvider

    store = CredentialStore(
        {"volcengine_speech": {"api_key": "speech-xapi-key"}}
    )
    provider = build_tts("volcengine_speech", store=store)
    assert isinstance(provider, VolcengineTTSProvider)
    assert provider._api_key == "speech-xapi-key"


def test_asr_partial_stable_s_default_and_override() -> None:
    """pipeline-latency-tail § A: ASR EOS endpoint defaults to 0.4 and is
    overridable per-campaign via build_asr(partial_stable_s=...)."""
    from isales_common.credentials import CredentialStore

    from isales_engine.providers.asr_volcengine import (
        DEFAULT_PARTIAL_STABLE_S,
        VolcengineASRProvider,
    )

    assert DEFAULT_PARTIAL_STABLE_S == 0.4
    assert VolcengineASRProvider(api_key="k")._partial_stable_s == 0.4

    store = CredentialStore({"volcengine_speech": {"api_key": "k"}})
    default_provider = build_asr("volcengine_speech", store=store)
    assert isinstance(default_provider, VolcengineASRProvider)
    assert default_provider._partial_stable_s == 0.4

    overridden = build_asr("volcengine_speech", store=store, partial_stable_s=0.25)
    assert isinstance(overridden, VolcengineASRProvider)
    assert overridden._partial_stable_s == 0.25


def test_factory_builds_dashscope_llm_from_store() -> None:
    """dashscope 是 LLM provider (OpenAI 兼容模式)；验证 store 装载路径不抛。"""
    from isales_common.credentials import CredentialStore

    store = CredentialStore(
        {
            "dashscope": {
                "api_key": "sk-dashscope-test",
            }
        }
    )
    provider = build_llm("dashscope", store=store)
    assert provider is not None


# ---- KeywordDrivenMockLLM --------------------------------------------------


def _msgs(system: str, user: str) -> list[Message]:
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


# Referee user-message primer (mirrors isales_engine.referee._USER_PRIMER).
_REF_PRIMER = "请按系统提示只输出一个分类词，不要任何其他内容、标点或 JSON。"


async def test_main_chat_stream_emits_plain_text() -> None:
    llm = KeywordDrivenMockLLM()
    out = [tok async for tok in llm.chat_stream(_msgs("你是销售助手。", "你好"))]
    text = "".join(out)
    assert text  # non-empty
    # plain text, NOT JSON
    assert not text.lstrip().startswith("{")
    assert llm.last_call_finish_reason == "stop"


async def test_main_short_reply_strategy() -> None:
    llm = KeywordDrivenMockLLM()
    out = [
        tok
        async for tok in llm.chat_stream(_msgs("你是销售助手。请用一句话回应", "用户内容"))
    ]
    assert "".join(out) == "明白了。"


async def test_main_wrap_up_segment_says_goodbye() -> None:
    llm = KeywordDrivenMockLLM()
    out = [
        tok
        async for tok in llm.chat_stream(_msgs("你是销售助手。目标已达成", "用户内容"))
    ]
    assert "再见" in "".join(out)


async def test_referee_continue_default() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("用户最后一句话：随便聊聊", _REF_PRIMER), json_mode=False)
    assert resp.content == "continue"  # bare token, no JSON


async def test_referee_appointment_goal_achieved() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(
        _msgs("用户最后一句话：我已经为您预约成功", _REF_PRIMER), json_mode=False
    )
    assert resp.content == "goal_achieved"


async def test_referee_transfer() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("用户最后一句话：我要转人工", _REF_PRIMER), json_mode=False)
    assert resp.content == "transfer"


async def test_referee_customer_decline() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("用户最后一句话：我不需要没兴趣", _REF_PRIMER), json_mode=False)
    assert resp.content == "customer_decline"


async def test_greeting_chat_returns_plain_text() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("你是销售助手。", "请生成开场白"))
    assert not resp.content.lstrip().startswith("{")
    assert resp.content


async def test_transfer_intent_keyword() -> None:
    llm = KeywordDrivenMockLLM()
    high = await llm.chat(_msgs("[transfer_intent]", "我要转人工"), json_mode=True)
    assert json.loads(high.content)["probability"] >= 0.9
    low = await llm.chat(_msgs("[transfer_intent]", "随便聊聊"), json_mode=True)
    assert json.loads(low.content)["probability"] < 0.5


async def test_transfer_llm_independent_marker() -> None:
    llm = KeywordDrivenMockLLM()
    yes = await llm.chat(_msgs("[transfer_llm]", "我要投诉"), json_mode=True)
    assert json.loads(yes.content)["transfer"] is True
    no = await llm.chat(_msgs("[transfer_llm]", "好的"), json_mode=True)
    assert json.loads(no.content)["transfer"] is False


# ---- ScriptedMockASR -------------------------------------------------------


async def test_scripted_asr_emits_partials_then_final() -> None:
    asr = ScriptedMockASR(partial_step_ms=10)

    async def empty_audio() -> asyncio.AsyncIterator[bytes]:  # type: ignore[name-defined]
        return
        yield b""  # unreachable; satisfies AsyncIterator type

    await asr.feed_turn("你好")
    await asr.close()

    results = [r async for r in asr.stream_recognize(empty_audio())]
    finals = [r for r in results if r.is_final]
    partials = [r for r in results if not r.is_final]
    assert len(partials) == 2  # "你", "你好"
    assert len(finals) >= 1
    assert finals[0].text == "你好"


# ---- TextLengthMockTTS -----------------------------------------------------


async def test_tts_emits_pcm_bytes_proportional_to_text() -> None:
    tts = TextLengthMockTTS(pcm_bytes_per_char=10, chunk_size=10)
    chunks = [c async for c in tts.synthesize_stream("hello", "voice-1")]
    total = sum(len(c) for c in chunks)
    assert total == len("hello") * 10
    assert tts.calls == [("hello", "voice-1")]
