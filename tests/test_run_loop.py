"""End-to-end smoke tests for run_session driving CallSession through the
full state machine + pipeline + filler + transfer + wrap-up loop using mock
providers + MockTelephonyClient. No DB / Redis needed for this layer — those
were exercised by PR #3 / PR #10."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from isales_common.providers._models import ASRResult
from isales_common.schemas.pipeline import ExtractorSpec, MainSpec, RefereeSpec

from isales_engine.call_session import CallSession
from isales_engine.pipeline.prompt_builder import (
    LeadInfo,
    PipelineConfig,
)
from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.filler_manager import FillerPhraseSpec, FillerSetSpec
from isales_engine.realtime.interruption_detector import InterruptionConfig
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.realtime.silence_detector import SilenceConfig
from isales_engine.run_loop import Providers, run_session
from isales_engine.runtime_config import RuntimeConfig
from isales_engine.transfer.manager import TransferConfig
from isales_engine.wrapup.manager import WrapUpConfig


def _make_session(call_id: int = 1) -> CallSession:
    return CallSession(
        call_record_id=call_id,
        campaign_id=10,
        lead_id=5,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


def _make_config(
    *,
    transfer_keywords: tuple[str, ...] = (),
    transfer_llm_enabled: bool = False,
    wrap_up_max_rounds: int = 1,
    wrap_up_max_seconds: int = 30,
    silence_threshold_ms: int = 500,
    filler_enabled: bool = False,
    # Legacy params kept for call-site compatibility; the dual-LLM pipeline has
    # exactly one main / referee / extractor slot (no N-role / M-judge PK).
    n_roles: int = 1,
    n_judges: int = 0,
) -> RuntimeConfig:
    pipeline = PipelineConfig(
        main=MainSpec(
            role_config_id=100, prompt_version_id=200, system_prompt="你是销售助手。"
        ),
        referee=RefereeSpec(
            role_config_id=300,
            prompt_version_id=400,
            system_prompt=(
                "判定对话状态。用户最后一句话：{{user_last_utterance}}\n"
                "最近对话：\n{{recent_dialog_history}}"
            ),
        ),
        extractor=ExtractorSpec(
            role_config_id=500, prompt_version_id=600, system_prompt="抽取字段。"
        ),
        default_replies=["好的，请稍等。"],
        lead=LeadInfo(name="李四", phone="+8613800000000", custom_data={}),
    )
    return RuntimeConfig(
        pipeline=pipeline,
        fillers=[
            FillerSetSpec(
                id=1,
                sort_order=0,
                phrases=[
                    FillerPhraseSpec(
                        id=11,
                        text="嗯",
                        audio_url="oss://x/11.wav",
                        generation_status="ready",
                    )
                ],
            )
        ],
        transfer=TransferConfig(
            keyword_enabled=bool(transfer_keywords),
            keywords=transfer_keywords,
            intent_enabled=False,
            intent_threshold=None,
            intent_system_prompt="",
            round_enabled=False,
            round_threshold=None,
            llm_enabled=transfer_llm_enabled,
            llm_system_prompt="独立判定是否转人工。",
            phrases=("您稍等，专员稍后联系您。",),
        ),
        wrap_up=WrapUpConfig(
            max_rounds=wrap_up_max_rounds,
            max_seconds=wrap_up_max_seconds,
            closing_phrases=("好的，再见。",),
        ),
        interruption=InterruptionConfig(
            whitelist=("嗯", "好的"), min_duration_ms=400
        ),
        silence=SilenceConfig(
            threshold_ms=silence_threshold_ms,
            max_activations=1,
            phrases=("您还在吗？",),
            hangup_phrase="再见。",
        ),
        voice_id="default",
        fixed_greeting="您好我是 AI 助手。",
        max_no_progress_seconds=None,
        filler_enabled=filler_enabled,
    )


def _make_providers(asr: ScriptedMockASR | None = None) -> Providers:
    return Providers(
        llm=KeywordDrivenMockLLM(),
        asr=asr or ScriptedMockASR(partial_step_ms=10),
        tts=TextLengthMockTTS(pcm_bytes_per_char=10, chunk_size=10),
    )


# ---- happy path: 1-turn call ending in user remote hangup ------------------


async def test_run_session_1_turn_then_remote_hangup() -> None:
    session = _make_session()
    config = _make_config()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        # User speaks 1 turn after greeting completes; then remote hangs up.
        await asyncio.sleep(0.05)
        await asr.feed_turn("你好这是用户的回复内容")
        await asyncio.sleep(0.15)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await driver_task

    assert session.state.value == "end"
    assert session.hangup_cause == "user_hangup"
    types = [e["type"] for e in session.full_transcript]
    assert "greeting" in types
    assert "user_speech" in types
    assert "ai_reply" in types
    assert "hangup" in types
    # pipeline_trace recorded for the 1 PROCESSING turn.
    assert len(session.pipeline_trace_records) == 1


# ---- transfer keyword hits → marked_for_handoff ---------------------------


async def test_run_session_transfer_keyword_marks_handoff() -> None:
    session = _make_session()
    config = _make_config(transfer_keywords=("转人工",))
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("我要转人工服务")
        await asyncio.sleep(0.05)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await driver_task

    assert session.state.value == "end"
    assert session.hangup_cause == "marked_for_handoff"
    assert session.transfer_status == "marked_for_handoff"
    assert session.transfer_reason == "keyword"
    types = [e["type"] for e in session.full_transcript]
    assert "transfer_initiated" in types
    assert "transfer_marked" in types
    # The pipeline should NOT have run for a transfer turn.
    assert session.pipeline_trace_records == []


async def test_run_session_transfer_llm_parallel_marks_handoff() -> None:
    """pipeline-latency-tail § D: transfer_llm runs in parallel with main
    streaming (not before PROCESSING). The main pipeline still runs that turn,
    and the parallel detector drives the handoff when the referee did not."""
    session = _make_session()
    # "投诉" trips the mock independent transfer_llm (transfer=true) but NOT the
    # mock referee (its 转人工/人工 regex misses 投诉 → decision=continue), so the
    # parallel detector is what marks the handoff.
    config = _make_config(transfer_llm_enabled=True, silence_threshold_ms=3000)
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("我要投诉你们的服务")
        await asyncio.sleep(0.15)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await driver_task

    assert session.state.value == "end"
    assert session.transfer_status == "marked_for_handoff"
    assert session.transfer_reason == "llm"
    # The main pipeline DID run this turn — transfer_llm did not gate PROCESSING.
    assert session.pipeline_trace_records != []


# ---- goal_achieved → wrap-up → exhausted → END ----------------------------


async def test_run_session_goal_achieved_enters_wrap_up_then_exhausts() -> None:
    session = _make_session()
    # max_rounds=1: after the first wrap-up round we should hangup.
    # Generous silence_threshold so the test driver can feed both turns
    # without silence firing first.
    config = _make_config(wrap_up_max_rounds=1, silence_threshold_ms=3000)
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        # Mock LLM: keyword "预约" → goal_achieved=true.
        await asr.feed_turn("我想预约一下")
        await asyncio.sleep(0.15)
        # Wrap-up round 1: any further user input runs simplified pipeline,
        # then evaluate_wrap_up fires (max_rounds=1) → END.
        await asr.feed_turn("好的明白")
        await asyncio.sleep(0.15)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await driver_task

    assert session.state.value == "end"
    assert session.hangup_cause == "wrap_up_completed"
    types = [e["type"] for e in session.full_transcript]
    assert "wrap_up_started" in types
    assert "goal_achieved" in types
    assert "wrap_up_completed" in types
    # 1 main pipeline + 1 wrap-up simplified pipeline = 2 traces.
    assert len(session.pipeline_trace_records) == 2


# ---- silence: no user speech → activation → still nothing → hangup --------


async def test_run_session_silence_activation_then_hangup() -> None:
    session = _make_session()
    config = _make_config(silence_threshold_ms=100)
    # ASR feeds nothing → silence triggers; max_activations=1 → second silence
    # tick triggers hangup.
    providers = _make_providers(asr=ScriptedMockASR(partial_step_ms=5))
    tel = MockTelephonyClient(connect_delay_ms=0)

    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )

    assert session.state.value == "end"
    assert session.hangup_cause == "silence_max_reached"
    types = [e["type"] for e in session.full_transcript]
    assert "silence_activation" in types


# ---- dial timeout / no connect → end with no_answer -----------------------


async def test_run_session_dial_no_connect_returns_no_answer() -> None:
    session = _make_session()
    config = _make_config()
    providers = _make_providers()

    class _NeverConnects(MockTelephonyClient):
        async def dial(self, call_id: int, phone: str) -> None:
            # Override: don't emit any event.
            self._ensure_call(call_id)

    tel = _NeverConnects(connect_delay_ms=0)
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
        connect_timeout_s=0.05,
    )

    assert session.state.value == "end"
    assert session.hangup_cause == "no_answer"


# Silence the unused-import linter for ASRResult (used in type hints in the
# orchestration tests above for clarity).
_ = ASRResult
_ = Any


pytestmark = pytest.mark.asyncio(loop_scope="session")
