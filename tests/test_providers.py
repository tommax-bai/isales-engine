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
    """LLM real providers raise NotImplementedError when credentials are
    missing (PR #1 wired the stub; PR #2/#3 added real impls)."""

    from isales_engine.settings import Settings

    empty = Settings(
        ISALES_DATABASE_URL="postgresql+asyncpg://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/0",
    )
    with pytest.raises(NotImplementedError):
        build_llm("openai", settings=empty)
    with pytest.raises(NotImplementedError):
        build_asr("volcengine", settings=empty)
    with pytest.raises(NotImplementedError):
        build_tts("alibaba")


# ---- KeywordDrivenMockLLM --------------------------------------------------


def _msgs(system: str, user: str) -> list[Message]:
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


async def test_role_default_returns_parsable_json_in_json_mode() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("[role]", "你好"), json_mode=True)
    parsed = json.loads(resp.content)
    assert parsed["goal_achieved"] is False
    assert parsed["goal_type"] == ""


async def test_role_default_emits_explanatory_chatter_outside_json_mode() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("[role]", "你好"), json_mode=False)
    # Surrounding text drives the regex fallback in PR #6's json_parser.
    assert "解释" in resp.content
    assert "{" in resp.content and "}" in resp.content


async def test_role_appointment_keyword_marks_goal_achieved() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(
        _msgs("[role]", "我已经为您预约成功"), json_mode=True
    )
    parsed = json.loads(resp.content)
    assert parsed["goal_achieved"] is True
    assert parsed["goal_type"] == "appointment"
    assert "appointment_time" in parsed["extracted"]


async def test_role_short_reply_strategy() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(
        _msgs("[role] 请用一句话回应", "用户内容"), json_mode=True
    )
    parsed = json.loads(resp.content)
    assert parsed["reply"] == "明白了。"


async def test_role_wrap_up_segment_does_not_re_trigger_goal_achieved() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(
        _msgs("[role] 目标已达成", "用户内容"), json_mode=True
    )
    parsed = json.loads(resp.content)
    assert parsed["goal_achieved"] is False


async def test_role_do_not_call_marker() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("[role]", "内部信号 do_not_call"), json_mode=True)
    parsed = json.loads(resp.content)
    assert parsed["goal_type"] == "do_not_call"
    assert parsed["goal_achieved"] is True


async def test_judge_default_passes() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(_msgs("[judge]", "好的，请稍等"), json_mode=True)
    parsed = json.loads(resp.content)
    assert parsed["passed"] is True


async def test_judge_reject_marker() -> None:
    llm = KeywordDrivenMockLLM()
    resp = await llm.chat(
        _msgs("[judge]", "**reject** content"), json_mode=True
    )
    parsed = json.loads(resp.content)
    assert parsed["passed"] is False


async def test_polish_picks_first_candidate_and_prepends() -> None:
    llm = KeywordDrivenMockLLM()
    user = "candidate[0]: 您好\ncandidate[1]: 您好啊"
    resp = await llm.chat(_msgs("[polish]", user), json_mode=True)
    parsed = json.loads(resp.content)
    assert parsed["selected_candidate_index"] == 0
    assert parsed["reply"] == "好的，您好"


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
