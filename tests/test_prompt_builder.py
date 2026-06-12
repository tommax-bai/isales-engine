"""Tests for main-LLM message assembly (engine-main-native-multiturn).

Locks in the native multi-turn contract: ``build_main_messages`` returns one
system message + ``dialog_history`` mapped 1:1 to native user/assistant turns,
with session-static context (summary + lead) living in the system message and
NO flattened transcript / trailing "AI:" hook.
"""

from __future__ import annotations

from isales_common.schemas.pipeline import ExtractorSpec, MainSpec, RefereeSpec

from isales_engine.call_session import CallSession, DialogTurn
from isales_engine.pipeline.prompt_builder import (
    LeadInfo,
    PipelineConfig,
    build_main_messages,
)


def _session() -> CallSession:
    return CallSession(
        call_record_id=1,
        campaign_id=1,
        lead_id=1,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


def _config(**over) -> PipelineConfig:
    base = {
        "main": MainSpec(role_config_id=1, prompt_version_id=1, system_prompt="你是助手。"),
        "referees": [
            RefereeSpec(role_config_id=2, prompt_version_id=2, system_prompt="判定。", label="j")
        ],
        "extractor": ExtractorSpec(role_config_id=3, prompt_version_id=3, system_prompt="抽取。"),
        "default_replies": ["好的。"],
        "lead": LeadInfo(name="李四", phone="+86138", custom_data={}),
    }
    base.update(over)
    return PipelineConfig(**base)


def test_dialog_history_maps_to_native_multiturn() -> None:
    session = _session()
    session.dialog_history = [
        DialogTurn(role="assistant", text="您好，我是顾问。", ts_ms=0),
        DialogTurn(role="user", text="你说吧", ts_ms=1),
        DialogTurn(role="assistant", text="我们有个课程。", ts_ms=2),
        DialogTurn(role="user", text="多少钱", ts_ms=3),
    ]
    msgs = build_main_messages(session, _config(), is_wrap_up=False)

    # system + 4 dialog turns, role-for-role, content-for-content, in order.
    assert [m.role for m in msgs] == ["system", "assistant", "user", "assistant", "user"]
    assert [m.content for m in msgs[1:]] == [
        "您好，我是顾问。",
        "你说吧",
        "我们有个课程。",
        "多少钱",
    ]


def test_no_flattened_transcript_or_ai_hook() -> None:
    session = _session()
    session.dialog_history = [
        DialogTurn(role="assistant", text="开场白。", ts_ms=0),
        DialogTurn(role="user", text="嗯", ts_ms=1),
    ]
    msgs = build_main_messages(session, _config(), is_wrap_up=False)

    # No single user message carries the whole transcript; no "AI:" hook prefix.
    joined = "\n".join(m.content for m in msgs)
    assert "【对话】" not in joined
    assert "AI:" not in joined
    assert "用户:" not in joined
    # The latest user turn is the last message (model generates next assistant).
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "嗯"


def test_lead_info_in_system_not_dialogue() -> None:
    session = _session()
    session.dialog_history = [DialogTurn(role="user", text="你好", ts_ms=0)]
    msgs = build_main_messages(session, _config(), is_wrap_up=False)

    system = msgs[0].content
    assert msgs[0].role == "system"
    assert "你是助手。" in system
    assert "【线索信息】" in system and "李四" in system
    # Lead info must NOT leak into any dialogue turn.
    assert all("【线索信息】" not in m.content for m in msgs[1:])


def test_follow_up_summary_in_system() -> None:
    session = _session()
    session.dialog_history = [DialogTurn(role="user", text="喂", ts_ms=0)]
    config = _config(last_call_summary="上次聊过价格。", follow_up_count=2)
    msgs = build_main_messages(session, config, is_wrap_up=False)

    system = msgs[0].content
    assert "【上次通话纪要】" in system and "上次聊过价格。" in system
    assert "【跟进上下文】" in system
    # Summary must be above the follow-up note that references "上述【上次通话纪要】".
    assert system.index("【上次通话纪要】") < system.index("【跟进上下文】")


def test_wrap_up_append_is_last() -> None:
    session = _session()
    session.dialog_history = [DialogTurn(role="user", text="好", ts_ms=0)]
    msgs = build_main_messages(session, _config(), is_wrap_up=True)

    system = msgs[0].content
    assert "收尾对话" in system
    # WRAP_UP_APPEND MUST be the literal tail of the system message.
    assert system.index("收尾对话") > system.index("【线索信息】")
    assert system.rstrip().endswith("不要再尝试推进新议题。")


def test_empty_dialog_history_yields_system_only() -> None:
    session = _session()  # dialog_history empty (defensive; main runs after greeting)
    msgs = build_main_messages(session, _config(), is_wrap_up=False)
    assert len(msgs) == 1
    assert msgs[0].role == "system"
