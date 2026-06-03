"""Tests for the referee module (pipeline-stream-and-referee)."""

from __future__ import annotations

import json

import pytest
from isales_common.providers._models import LLMResponse
from isales_common.providers.testing.llm import MockLLMProvider
from isales_common.schemas.pipeline import RefereeSpec

from isales_engine.call_session import DialogTurn
from isales_engine.referee import (
    _EMPTY_HISTORY_PLACEHOLDER,
    _render_dialog_history_for_referee,
    recent_dialog_rounds,
    run_referee,
)


def _spec() -> RefereeSpec:
    return RefereeSpec(
        role_config_id=2,
        prompt_version_id=8,
        system_prompt=(
            "用户最后一句话：{{user_last_utterance}}\n"
            "最近对话：\n{{recent_dialog_history}}\n输出 JSON。"
        ),
        model="qwen-turbo",
    )


def _llm(payload: dict) -> MockLLMProvider:
    return MockLLMProvider(
        responses=[
            LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                tokens_in=10,
                tokens_out=5,
                finish_reason="stop",
                latency_ms=0,
            )
        ]
    )


def _raw_llm(content: str) -> MockLLMProvider:
    return MockLLMProvider(
        responses=[
            LLMResponse(
                content=content, tokens_in=1, tokens_out=1, finish_reason="stop", latency_ms=0
            )
        ]
    )


# ---- dialog history rendering ---------------------------------------------


def test_render_empty_history_placeholder():
    assert _render_dialog_history_for_referee([]) == _EMPTY_HISTORY_PLACEHOLDER


def test_render_history_full_width_colon():
    turns = [
        DialogTurn(role="assistant", text="您好", ts_ms=0),
        DialogTurn(role="user", text="你好", ts_ms=1),
    ]
    out = _render_dialog_history_for_referee(turns)
    assert out == "AI：您好\n用户：你好"


def test_recent_dialog_rounds_caps_at_3_rounds():
    turns = [DialogTurn(role="user", text=str(i), ts_ms=i) for i in range(20)]
    recent = recent_dialog_rounds(turns, rounds=3)
    assert len(recent) == 6
    assert recent[-1].text == "19"


# ---- decision enums --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected_decision,expected_goal",
    [
        ({"decision": "continue", "goal_type": None, "confidence": 0.9}, "continue", None),
        (
            {"decision": "goal_achieved", "goal_type": "appointment", "confidence": 0.95},
            "goal_achieved",
            "appointment",
        ),
        (
            {"decision": "customer_decline", "goal_type": None, "confidence": 0.88},
            "customer_decline",
            None,
        ),
        ({"decision": "transfer", "goal_type": None, "confidence": 0.9}, "transfer", None),
    ],
)
async def test_four_decisions(payload, expected_decision, expected_goal):
    result = await run_referee(None, "好的", [], _spec(), _llm(payload))
    assert result.decision == expected_decision
    assert result.goal_type == expected_goal


# ---- fail-open paths -------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_open_on_invalid_json():
    result = await run_referee(None, "x", [], _spec(), _raw_llm("not json at all"))
    assert result.decision == "continue"
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_fail_open_on_unknown_decision():
    result = await run_referee(
        None, "x", [], _spec(), _llm({"decision": "explode", "confidence": 0.9})
    )
    assert result.decision == "continue"


@pytest.mark.asyncio
async def test_low_confidence_downgraded_to_continue():
    result = await run_referee(
        None,
        "x",
        [],
        _spec(),
        _llm({"decision": "goal_achieved", "goal_type": "sale", "confidence": 0.5}),
    )
    assert result.decision == "continue"
    assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_goal_achieved_without_goal_type_downgrades():
    result = await run_referee(
        None, "x", [], _spec(), _llm({"decision": "goal_achieved", "confidence": 0.95})
    )
    assert result.decision == "continue"


@pytest.mark.asyncio
async def test_fail_open_on_llm_exception():
    class BoomLLM(MockLLMProvider):
        async def chat(self, *a, **k):
            raise RuntimeError("provider down")

    result = await run_referee(None, "x", [], _spec(), BoomLLM())
    assert result.decision == "continue"


@pytest.mark.asyncio
async def test_placeholder_substituted_into_prompt():
    llm = _llm({"decision": "continue", "confidence": 0.9})
    await run_referee(None, "周三方便", [DialogTurn(role="user", text="嗯", ts_ms=0)], _spec(), llm)
    system_msg = llm.calls[0].messages[0].content
    assert "周三方便" in system_msg
    assert "{{user_last_utterance}}" not in system_msg
    assert "用户：嗯" in system_msg
