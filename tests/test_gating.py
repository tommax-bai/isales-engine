"""Pre-reply referee gating + lazy tool routes + eager personas (gate-first).

engine-tools-multidialogue-gating §7.4 / §7.5 / §7.6. These drive run_session
with ENGINE_USE_ROUTER ON and assert the gate-first behaviors the golden net
deliberately does not pin (the golden blanks the gating columns + shares one
fixture across flags for byte-stable scenarios).
"""

from __future__ import annotations

import asyncio

import pytest
from isales_common.schemas.pipeline import PersonaSpec

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import Providers, run_session
from tests.test_run_loop import _make_config, _make_providers, _make_session

pytestmark = pytest.mark.asyncio


def _persona(label: str, rcid: int) -> PersonaSpec:
    return PersonaSpec(
        role_config_id=rcid,
        prompt_version_id=rcid + 1,
        system_prompt=f"{label} 风格回复。",
        label=label,
    )


async def _run(config, *, turns: tuple[str, ...], hangup_after: bool = False):
    session = _make_session()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        for t in turns:
            await asr.feed_turn(t)
            await asyncio.sleep(0.15)
        if hangup_after:
            await tel.simulate_remote_hangup(1)

    task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await task
    return session


# --------------------------------------------------------------------------- #
# Lazy tool routes
# --------------------------------------------------------------------------- #


async def test_tool_hangup_suppresses_reply_and_ends_with_referee_hangup() -> None:
    config = _make_config(
        engine_use_router=True,
        routing_rules=[
            {
                "referee": "main_judge",
                "match": ["customer_decline"],
                "action": {"type": "tool", "tool": "bye"},
            }
        ],
    )
    config.pipeline.tools = {
        "bye": {"type": "hangup", "closing_phrase": "祝您生活愉快，再见。", "interrupt": False}
    }
    # "不需要" → mock referee category customer_decline → tool:bye (hangup).
    session = await _run(config, turns=("我不需要了",))

    assert session.state.value == "end"
    assert session.hangup_cause == "referee_hangup"
    types = [e["type"] for e in session.full_transcript]
    assert "ai_reply" not in types  # dialogue reply suppressed
    assert any(
        e["type"] == "hangup" and e["reason"] == "referee_hangup"
        for e in session.full_transcript
    )
    trace = session.pipeline_trace_records[-1]
    assert trace["selected_route_id"] == "tool:bye"
    assert trace["selected_route_kind"] == "tool"


async def test_tool_transfer_routes_through_perform_handoff() -> None:
    config = _make_config(
        engine_use_router=True,
        routing_rules=[
            {
                "referee": "main_judge",
                "match": ["transfer"],
                "action": {"type": "tool", "tool": "agent"},
            }
        ],
    )
    config.pipeline.tools = {"agent": {"type": "transfer"}}
    # "转人工" → mock referee category transfer → tool:agent (transfer handoff).
    session = await _run(config, turns=("我要转人工",))

    assert session.state.value == "end"
    assert session.hangup_cause == "marked_for_handoff"
    assert session.transfer_status == "marked_for_handoff"
    types = [e["type"] for e in session.full_transcript]
    assert "transfer_initiated" in types
    assert "transfer_marked" in types
    assert "ai_reply" not in types  # reply suppressed; transfer replaces it
    trace = session.pipeline_trace_records[-1]
    assert trace["selected_route_id"] == "tool:agent"  # route id is tool:<alias>
    assert trace["selected_route_kind"] == "tool"


async def test_unknown_tool_alias_fails_open_to_main() -> None:
    config = _make_config(
        engine_use_router=True,
        routing_rules=[
            {
                "referee": "main_judge",
                "match": ["customer_decline"],
                "action": {"type": "tool", "tool": "does_not_exist"},
            }
        ],
    )
    config.pipeline.tools = {}  # alias missing → runtime fail-open to main
    session = await _run(config, turns=("我不需要了",), hangup_after=True)

    # Did NOT crash / hang up via referee — fell open to the main dialogue route.
    assert session.hangup_cause == "user_hangup"
    trace = session.pipeline_trace_records[-1]
    assert trace["selected_route_id"] == "main"
    assert trace["selected_route_kind"] == "dialogue"


# --------------------------------------------------------------------------- #
# Eager speculative personas
# --------------------------------------------------------------------------- #


async def test_eager_persona_selected_one_cancel_rest() -> None:
    config = _make_config(
        engine_use_router=True,
        routing_rules=[
            {
                "referee": "main_judge",
                "match": ["customer_decline"],
                "action": {"type": "route", "to": "empathetic"},
            }
        ],
    )
    config.pipeline.personas = [_persona("empathetic", 900)]
    config.pipeline.persona_fanout_cap = 2  # main + 1 persona speculated
    session = await _run(config, turns=("我不需要了",), hangup_after=True)

    trace = session.pipeline_trace_records[-1]
    assert trace["persona_candidates"] == ["main", "persona:empathetic"]
    assert trace["selected_route_id"] == "persona:empathetic"
    assert trace["selected_route_kind"] == "dialogue"
    # The persona reply was played (winner released).
    assert any(e["type"] == "ai_reply" for e in session.full_transcript)


async def test_persona_fanout_cap_clamped_to_three() -> None:
    config = _make_config(engine_use_router=True)  # no rule → continue → main
    config.pipeline.personas = [_persona(f"p{i}", 900 + i * 2) for i in range(4)]
    config.pipeline.persona_fanout_cap = 5  # clamp to 3 (main + 2 personas)
    session = await _run(config, turns=("随便聊聊",), hangup_after=True)

    trace = session.pipeline_trace_records[-1]
    assert len(trace["persona_candidates"]) == 3
    assert trace["persona_candidates"][0] == "main"
    # continue (no rule) → fail open to main even with personas speculated.
    assert trace["selected_route_id"] == "main"


# --------------------------------------------------------------------------- #
# Gate fail-open on referee timeout
# --------------------------------------------------------------------------- #


async def test_gate_fails_open_to_main_on_referee_timeout() -> None:
    class _SlowRefereeLLM(KeywordDrivenMockLLM):
        async def chat(self, messages, *, json_mode=False, **kw):
            user = self._first_role(messages, "user")
            if "JSON schema 输出" in user:  # referee call → stall past the gate
                await asyncio.sleep(0.2)
            return await super().chat(messages, json_mode=json_mode, **kw)

    config = _make_config(
        engine_use_router=True,
        routing_rules=[
            {
                "referee": "main_judge",
                "match": ["customer_decline"],
                "action": {"type": "tool", "tool": "bye"},
            }
        ],
    )
    config.pipeline.tools = {"bye": {"type": "hangup"}}
    config.pipeline.referee_timeout_ms = 20  # << the 0.2s referee stall

    session = _make_session()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = Providers(
        llm=_SlowRefereeLLM(),
        asr=asr,
        tts=_make_providers().tts,
    )
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("我不需要了")  # would map to hangup IF the referee landed
        await asyncio.sleep(0.2)
        await tel.simulate_remote_hangup(1)

    task = asyncio.create_task(driver())
    await run_session(
        session, phone="+8613800000000", config=config, telephony=tel, providers=providers
    )
    await task

    # Referee timed out → no rule could match → fail open to main, NOT hangup.
    assert session.hangup_cause == "user_hangup"
    trace = session.pipeline_trace_records[-1]
    assert trace["selected_route_id"] == "main"
    # The fail-open referee is recorded with the "timeout" marker category.
    assert any(r["category"] == "timeout" for r in trace["referee_results"])
