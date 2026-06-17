"""End-to-end smoke tests for run_session driving CallSession through the
full state machine + pipeline + filler + transfer + wrap-up loop using mock
providers + MockTelephonyClient. No DB / Redis needed for this layer — those
were exercised by PR #3 / PR #10."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from isales_common.providers._models import ASRResult
from isales_common.schemas.pipeline import (
    ExtractorSpec,
    MainSpec,
    RefereeSpec,
    RestructureSpec,
)

from isales_engine.call_session import CallSession, DialogTurn
from isales_engine.pipeline.prompt_builder import (
    LeadInfo,
    PipelineConfig,
)
from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.interruption_detector import InterruptionConfig
from isales_engine.realtime.interruption_rules import default_rule
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.realtime.silence_detector import SilenceConfig
from isales_engine.run_loop import (
    Providers,
    _assemble_interrupt_text,
    run_session,
)
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
    wrap_up_silence_hangup_ms: int = 6000,
    filler_enabled: bool = False,
    enable_restructure: bool = False,
    routing_rules: list | None = None,
    max_continuous_restructure: int = 2,
    auto_restructure_on_interrupt: bool = False,
    # Legacy params kept for call-site compatibility; the dual-LLM pipeline has
    # exactly one main / referee / extractor slot (no N-role / M-judge PK).
    n_roles: int = 1,
    n_judges: int = 0,
) -> RuntimeConfig:
    restructure_spec = (
        RestructureSpec(
            role_config_id=700,
            prompt_version_id=800,
            system_prompt="用口语化方式重组这句话：",
            label="rewrite",
        )
        if enable_restructure
        else None
    )
    pipeline = PipelineConfig(
        main=MainSpec(
            role_config_id=100, prompt_version_id=200, system_prompt="你是销售助手。"
        ),
        referees=[
            RefereeSpec(
                role_config_id=300,
                prompt_version_id=400,
                system_prompt=(
                    "判定对话状态。用户最后一句话：{{user_last_utterance}}\n"
                    "最近对话：\n{{recent_dialog_history}}"
                ),
                label="main_judge",
            )
        ],
        # Equivalent default routing rules seeded by the migration: the mock
        # referee emits goal_achieved / transfer / customer_decline categories.
        routing_rules=routing_rules
        if routing_rules is not None
        else [
            {
                "referee": "main_judge",
                "match": ["goal_achieved"],
                "action": {
                    "type": "transition",
                    "to": "goal_achieved",
                    "goal_type": "appointment",
                },
            },
            {
                "referee": "main_judge",
                "match": ["transfer"],
                "action": {"type": "transition", "to": "transfer"},
            },
            {
                "referee": "main_judge",
                "match": ["customer_decline"],
                "action": {"type": "transition", "to": "customer_decline"},
            },
        ],
        max_continuous_restructure=max_continuous_restructure,
        auto_restructure_on_interrupt=auto_restructure_on_interrupt,
        restructure=restructure_spec,
        extractor=ExtractorSpec(
            role_config_id=500, prompt_version_id=600, system_prompt="抽取字段。"
        ),
        default_replies=["好的，请稍等。"],
        lead=LeadInfo(name="李四", phone="+8613800000000", custom_data={}),
    )
    return RuntimeConfig(
        pipeline=pipeline,
        filler_phrases=["嗯"],
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
            whitelist=("嗯", "好的"),
            rule=default_rule(whitelist=["嗯", "好的"], min_duration_ms=400),
        ),
        silence=SilenceConfig(
            threshold_ms=silence_threshold_ms,
            max_activations=1,
            phrases=("您还在吗？",),
            hangup_phrase="再见。",
            wrap_up_hangup_ms=wrap_up_silence_hangup_ms,
        ),
        voice_id="default",
        fixed_greeting="您好我是 AI 助手。",
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


# ---- engine-wrap-up-silence-hangup: call-215 regression -------------------


async def test_run_session_wrap_up_silence_hangs_up_without_reactivation() -> None:
    # call-215: goal achieved → enter wrap-up → customer goes silent after the
    # farewell. The engine MUST hang up directly (reason=wrap_up_silence) and MUST
    # NOT replay the "您还在吗？" reactivation. wrap_up window (300ms) is set LONGER
    # than the main silence threshold (100ms) to also guard the asyncio.wait
    # timing fix (D3): if the wait woke at 100ms and never accumulated to 300ms,
    # the hangup would never fire and this test would hang/time out.
    session = _make_session()
    config = _make_config(
        silence_threshold_ms=100,
        wrap_up_silence_hangup_ms=300,
        wrap_up_max_rounds=5,  # high so the counter does NOT fire first
        wrap_up_max_seconds=30,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        # "预约" → goal_achieved → enters wrap-up. Then the customer says nothing.
        await asr.feed_turn("我想预约一下")
        # No further turns: silence in wrap-up should drive the hangup.

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
    assert session.hangup_cause == "silence_max_reached"
    types = [e["type"] for e in session.full_transcript]
    assert "wrap_up_started" in types  # confirms we entered wrap-up
    # The reactivation ladder MUST be skipped in wrap-up.
    assert "silence_activation" not in types
    # Hangup recorded under the distinct free-form reason, initiated by the AI.
    hangups = [e for e in session.full_transcript if e["type"] == "hangup"]
    assert len(hangups) == 1
    assert hangups[0]["reason"] == "wrap_up_silence"
    assert hangups[0]["initiated_by"] == "ai"
    # The counter-exhaustion path did NOT fire.
    assert "wrap_up_completed" not in types


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


async def test_run_session_silence_max_empty_phrase_direct_hangup() -> None:
    # §11: empty silence_hangup_phrase → direct hangup, no farewell played.
    session = _make_session()
    config = _make_config(silence_threshold_ms=100)
    config.silence = SilenceConfig(
        threshold_ms=100, max_activations=0, phrases=(), hangup_phrase=""
    )
    providers = _make_providers(asr=ScriptedMockASR(partial_step_ms=5))
    tel = MockTelephonyClient(connect_delay_ms=0)

    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )

    assert session.state.value == "end"
    assert session.hangup_cause == "silence_max_reached"
    spoken = [text for (text, _voice) in providers.tts.calls]
    assert "再见。" not in spoken  # old fallback farewell no longer played


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


# ---- restructure: InterruptText assembly (§4.4 / §4.5) --------------------


async def test_assemble_interrupt_text_last_reply():
    """source=last_reply takes the last assistant utterance (no history blob)."""
    sess = _make_session()
    sess.dialog_history.append(DialogTurn(role="assistant", text="第一句", ts_ms=0))
    sess.dialog_history.append(DialogTurn(role="user", text="嗯", ts_ms=1))
    sess.dialog_history.append(DialogTurn(role="assistant", text="最后一句要点", ts_ms=2))
    assert _assemble_interrupt_text(sess, "last_reply") == "最后一句要点"


async def test_assemble_interrupt_text_interrupt_remaining_consumes_and_clears():
    sess = _make_session()
    sess.dialog_history.append(DialogTurn(role="assistant", text="上一句", ts_ms=0))
    sess.interrupt_remaining_text = "被打断的残留内容"
    assert _assemble_interrupt_text(sess, "interrupt_remaining") == "被打断的残留内容"
    # consumed — cleared after taking it.
    assert sess.interrupt_remaining_text is None


async def test_assemble_interrupt_text_interrupt_remaining_degrades_when_empty():
    """Empty remainder degrades to last_reply (§4.5)."""
    sess = _make_session()
    sess.dialog_history.append(DialogTurn(role="assistant", text="退化到这句", ts_ms=0))
    sess.interrupt_remaining_text = None
    assert _assemble_interrupt_text(sess, "interrupt_remaining") == "退化到这句"


async def test_assemble_interrupt_text_none_when_no_assistant():
    sess = _make_session()
    sess.dialog_history.append(DialogTurn(role="user", text="只有用户", ts_ms=0))
    assert _assemble_interrupt_text(sess, "last_reply") is None


# ---- restructure: now triggered ONLY by the auto_restructure_on_interrupt
# switch (engine-auto-restructure-on-interrupt removed the restructure routing
# action; the old rule-driven fires/caps tests went with it). -----------------


async def test_auto_restructure_fires_when_gate_says_filler() -> None:
    """engine-filler-gated-restructure: when a prior barge-in left
    interrupt_remaining_text, NO explicit routing rule matches, AND the main gate
    referee judged the interruption a no-substance FILLER, the turn resumes the
    cut-off line (restructure source=interrupt_remaining) instead of replying."""
    session = _make_session()
    session.interrupt_remaining_text = "那我先帮您把套餐"  # leftover from a prior barge-in
    config = _make_config(
        enable_restructure=True,
        routing_rules=[],  # no explicit rule → decide() falls through to continue
        auto_restructure_on_interrupt=True,
        silence_threshold_ms=3000,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("嗯你继续")  # mock referee → FILLER (no-substance) → fires restructure
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await driver_task

    traces = session.pipeline_trace_records
    assert traces, "expected at least one PROCESSING trace"
    first = traces[0]
    assert first["restructure_active"] is True
    assert first["restructure_trigger"] == "interrupt_remaining"
    assert first["restructure_source_text"]  # the captured leftover was re-voiced
    assert session.consecutive_restructure_count >= 1

    # engine-turn-latency-and-tts-guard: the restructure OUTPUT turn now writes
    # its own pipeline_trace row too (was missing → call 194 turn-3 gap). So the
    # fired restructure produces a decision row + an output row with distinct
    # turn_ids, both restructure_active.
    restr = [t for t in traces if t["restructure_active"]]
    assert len(restr) == 2, "decision row + output row"
    assert len({t["turn_id"] for t in restr}) == 2, "distinct turn_ids"


async def test_auto_restructure_skipped_when_gate_says_substantive() -> None:
    """engine-filler-gated-restructure: a barge-in the gate judged substantive
    (non-FILLER, e.g. a price question) is answered normally, NOT restructured;
    and the stale remainder is cleared so it can't leak into a later turn."""
    session = _make_session()
    session.interrupt_remaining_text = "那我先帮您把套餐"  # leftover from a prior barge-in
    config = _make_config(
        enable_restructure=True,
        routing_rules=[],  # no explicit rule → decide() falls through to continue
        auto_restructure_on_interrupt=True,
        silence_threshold_ms=3000,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("这个一共要多少钱")  # mock referee → continue (substantive), not FILLER
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await driver_task

    # No restructure fired — substantive interruption is answered, not resumed.
    assert all(
        not t["restructure_active"] for t in session.pipeline_trace_records
    ), "substantive interruption must be answered, not restructured"
    # Cross-turn remainder hygiene: stale leftover cleared on the answer path (D6).
    assert session.interrupt_remaining_text is None


async def test_auto_restructure_vetoed_by_explicit_rule() -> None:
    """The auto fallback only takes the no-match slot: an explicit routing rule
    that matches the referee category wins (first-match-wins veto), so NO auto
    restructure even though interrupt_remaining_text is set."""
    session = _make_session()
    session.interrupt_remaining_text = "那我先帮您把套餐"
    veto_rule = [
        {
            "referee": "main_judge",
            "match": ["continue"],
            "action": {"type": "transition", "to": "customer_decline"},
        }
    ]
    config = _make_config(
        enable_restructure=True,
        routing_rules=veto_rule,
        auto_restructure_on_interrupt=True,
        silence_threshold_ms=3000,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("随便聊聊")  # mock referee → "continue", matches veto rule
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await driver_task

    assert all(
        not t["restructure_active"] for t in session.pipeline_trace_records
    ), "explicit rule must veto the auto restructure"


async def test_auto_restructure_noop_without_restructure_slot() -> None:
    """Switch ON but no restructure slot configured → fallback stays continue, no
    error, no restructure."""
    session = _make_session()
    session.interrupt_remaining_text = "那我先帮您把套餐"
    config = _make_config(
        enable_restructure=False,  # no restructure slot
        routing_rules=[],
        auto_restructure_on_interrupt=True,
        silence_threshold_ms=3000,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("嗯你继续")
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await driver_task

    assert all(not t.get("restructure_active") for t in session.pipeline_trace_records)


async def test_auto_restructure_noop_without_interrupt_remaining() -> None:
    """Switch ON but no barge-in this turn (interrupt_remaining_text empty) →
    fallback stays continue."""
    session = _make_session()  # interrupt_remaining_text stays None
    config = _make_config(
        enable_restructure=True,
        routing_rules=[],
        auto_restructure_on_interrupt=True,
        silence_threshold_ms=3000,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("嗯你继续")
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await driver_task

    assert all(not t["restructure_active"] for t in session.pipeline_trace_records)


async def test_auto_restructure_off_ignores_interrupt_remaining() -> None:
    """Switch OFF (default): even with interrupt_remaining_text set and no rule
    match, the turn falls through to continue — byte-for-byte unchanged."""
    session = _make_session()
    session.interrupt_remaining_text = "那我先帮您把套餐"
    config = _make_config(
        enable_restructure=True,
        routing_rules=[],
        auto_restructure_on_interrupt=False,
        silence_threshold_ms=3000,
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("嗯你继续")
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await driver_task

    assert all(not t["restructure_active"] for t in session.pipeline_trace_records)


# Silence the unused-import linter for ASRResult (used in type hints in the
# orchestration tests above for clarity).
_ = ASRResult
_ = Any


pytestmark = pytest.mark.asyncio(loop_scope="session")
