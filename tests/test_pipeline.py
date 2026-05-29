"""Tests for the AI pipeline orchestrator + json_parser + prompt_builder."""

from __future__ import annotations

import asyncio
import json

from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider

from isales_engine.call_session import CallSession
from isales_engine.pipeline.greeting import generate_greeting
from isales_engine.pipeline.json_parser import (
    parse_judge_output,
    parse_polish_output,
    parse_role_output,
)
from isales_engine.pipeline.orchestrator import PipelineOutcome, run_pipeline
from isales_engine.pipeline.prompt_builder import (
    JUDGE_OUTPUT_SCHEMA_SUFFIX,
    JudgeSpec,
    LeadInfo,
    PipelineConfig,
    PolishSpec,
    RoleSpec,
    build_judge_messages,
)
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM

# ---- json_parser ----------------------------------------------------------


def test_role_parser_strict_json() -> None:
    r = parse_role_output(
        '{"reply": "hi", "goal_achieved": false, "goal_type": "", "extracted": {}}'
    )
    assert not r.parse_failed
    assert r.reply == "hi"


def test_role_parser_regex_fallback() -> None:
    raw = (
        '前言... {"reply": "你好", "goal_achieved": true, '
        '"goal_type": "appointment", "extracted": {"x": 1}} 后续'
    )
    r = parse_role_output(raw)
    assert not r.parse_failed
    assert r.reply == "你好"
    assert r.goal_achieved is True
    assert r.extracted == {"x": 1}


def test_role_parser_failure_marks_parse_failed() -> None:
    assert parse_role_output("not json at all").parse_failed
    assert parse_role_output("").parse_failed
    # Empty reply → counted as parse failure (no usable content).
    assert parse_role_output('{"reply": ""}').parse_failed


def test_judge_parser_safe_default() -> None:
    assert parse_judge_output("garbage")[0] is False
    assert parse_judge_output('{"passed": true, "reason": "ok"}') == (True, "ok")
    assert parse_judge_output('{"passed": false, "reason": "no"}')[0] is False


def test_polish_parser_returns_none_on_failure() -> None:
    assert parse_polish_output("garbage") == (None, None)
    reply, idx = parse_polish_output(
        '{"reply": "hi", "selected_candidate_index": 1}'
    )
    assert reply == "hi"
    assert idx == 1


# ---- orchestrator helpers --------------------------------------------------


def _make_session() -> CallSession:
    return CallSession(
        call_record_id=1,
        campaign_id=10,
        lead_id=5,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


def _make_config(*, n_roles: int = 2, n_judges: int = 1) -> PipelineConfig:
    return PipelineConfig(
        roles=[
            RoleSpec(
                role_config_id=100 + i,
                prompt_version_id=200 + i,
                system_prompt=f"role-{i}",
            )
            for i in range(n_roles)
        ],
        judges=[
            JudgeSpec(
                role_config_id=300 + i,
                prompt_version_id=400 + i,
                system_prompt=f"judge-{i}",
            )
            for i in range(n_judges)
        ],
        polish=PolishSpec(
            role_config_id=999, prompt_version_id=1999, system_prompt="polish"
        ),
        default_replies=["好的，请稍等。"],
        lead=LeadInfo(name="李四", phone="+8613800000000", custom_data={"city": "上海"}),
    )


# ---- happy path ------------------------------------------------------------


async def test_pipeline_happy_path_polished_reply() -> None:
    session = _make_session()
    session.append_event("greeting", text="您好")
    session.append_event("user_speech", text="你好")
    config = _make_config()
    llm = KeywordDrivenMockLLM()

    outcome = await run_pipeline(session, "你好", config, llm)
    assert isinstance(outcome, PipelineOutcome)
    assert outcome.source == "polished"
    assert outcome.reply.startswith("好的，")
    # default keyword-driven mock => goal_achieved False, goal_type ""
    assert outcome.goal_achieved is False

    # pipeline_trace appended exactly once.
    assert len(session.pipeline_trace_records) == 1
    trace = session.pipeline_trace_records[0]
    assert trace["turn_id"] == 1
    assert len(trace["role_candidates"]) == 2
    assert len(trace["judge_results"]) == 2  # 2 candidates × 1 judge
    assert trace["final_selected_candidate_index"] in {0, 1}


async def test_pipeline_appointment_keyword_propagates_goal_achieved() -> None:
    session = _make_session()
    session.append_event("user_speech", text="我想预约")
    config = _make_config()
    llm = KeywordDrivenMockLLM()
    outcome = await run_pipeline(session, "我想预约成功", config, llm)
    assert outcome.goal_achieved is True
    assert outcome.goal_type == "appointment"
    assert "appointment_time" in outcome.extracted


# ---- judge rejection -------------------------------------------------------


class _ScriptedLLM(LLMProvider):
    """Returns scripted responses keyed by which '[layer]' tag is in system."""

    def __init__(self, *, role: str, judge: str, polish: str) -> None:
        self.role = role
        self.judge = judge
        self.polish = polish
        self.calls: list[Message] = []

    async def chat(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.extend(messages)
        system = next((m.content for m in messages if m.role == "system"), "")
        if "[judge]" in system:
            content = self.judge
        elif "[polish]" in system:
            content = self.polish
        else:
            content = self.role
        return LLMResponse(
            content=content,
            tokens_in=10,
            tokens_out=10,
            finish_reason="stop",
            latency_ms=0,
        )


async def test_pipeline_all_judges_reject_falls_back_to_default_reply() -> None:
    session = _make_session()
    session.append_event("user_speech", text="嗯")
    config = _make_config()
    llm = _ScriptedLLM(
        role=json.dumps(
            {
                "reply": "OK",
                "goal_achieved": False,
                "goal_type": "",
                "extracted": {},
            }
        ),
        judge='{"passed": false, "reason": "block"}',
        polish='{"reply": "x", "selected_candidate_index": 0}',
    )
    outcome = await run_pipeline(session, "你好", config, llm)
    assert outcome.source == "default_reply"
    assert outcome.error == "all_judges_rejected"
    assert any(
        e["type"] == "default_reply_used" for e in session.full_transcript
    )


async def test_pipeline_all_roles_parse_fail_falls_back() -> None:
    session = _make_session()
    config = _make_config()
    llm = _ScriptedLLM(
        role="not json",
        judge='{"passed": true, "reason": ""}',
        polish='{"reply": "x", "selected_candidate_index": 0}',
    )
    outcome = await run_pipeline(session, "你好", config, llm)
    assert outcome.source == "default_reply"
    assert outcome.error == "all_roles_failed_or_parse"


# ---- polish failure / fallback --------------------------------------------


async def test_pipeline_polish_failure_picks_first_passing_candidate() -> None:
    session = _make_session()
    config = _make_config()
    llm = _ScriptedLLM(
        role=json.dumps(
            {
                "reply": "candidate-reply",
                "goal_achieved": True,
                "goal_type": "appointment",
                "extracted": {"k": 1},
            }
        ),
        judge='{"passed": true, "reason": "ok"}',
        polish="garbage",
    )
    outcome = await run_pipeline(session, "你好", config, llm)
    assert outcome.source == "polish_fallback"
    assert outcome.reply == "candidate-reply"
    # Goal markers inherited from selected candidate (NOT from polish).
    assert outcome.goal_achieved is True
    assert outcome.goal_type == "appointment"
    assert outcome.extracted == {"k": 1}


async def test_pipeline_goal_markers_not_voted_across_candidates() -> None:
    """Even if multiple candidates have different goal_achieved values, only
    the SELECTED candidate's marker propagates to the outcome."""

    session = _make_session()
    # Both candidates pass the judge (default keyword-driven mock judge passes
    # everything). The polish picks index 0 (KeywordDrivenMockLLM polish always
    # selects [0]). Set up roles so candidate 0 has goal=False, candidate 1
    # has goal=True. Final outcome must reflect candidate 0 (goal=False).
    config = _make_config(n_roles=2, n_judges=1)
    seq = [
        json.dumps(
            {"reply": "no-goal", "goal_achieved": False, "goal_type": "", "extracted": {}}
        ),
        json.dumps(
            {
                "reply": "with-goal",
                "goal_achieved": True,
                "goal_type": "appointment",
                "extracted": {"x": 1},
            }
        ),
    ]
    judge_resp = '{"passed": true, "reason": "ok"}'
    polish_resp = '{"reply": "好的，no-goal", "selected_candidate_index": 0}'

    class _Seq(LLMProvider):
        def __init__(self) -> None:
            self.role_idx = 0

        async def chat(  # type: ignore[override]
            self,
            messages: list[Message],
            *,
            json_mode: bool = False,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            system = next((m.content for m in messages if m.role == "system"), "")
            if "[role]" in system:
                content = seq[self.role_idx % len(seq)]
                self.role_idx += 1
            elif "[judge]" in system:
                content = judge_resp
            else:
                content = polish_resp
            return LLMResponse(
                content=content, tokens_in=0, tokens_out=0, finish_reason="stop", latency_ms=0
            )

    llm = _Seq()
    outcome = await run_pipeline(session, "你好", config, llm)
    assert outcome.source == "polished"
    assert outcome.selected_candidate_index == 0
    # Polish picked index 0 → goal_achieved should be False (candidate 0).
    assert outcome.goal_achieved is False


# ---- pipeline_trace persistence in all paths ------------------------------


async def test_pipeline_trace_written_on_default_reply_path() -> None:
    session = _make_session()
    config = _make_config()
    llm = _ScriptedLLM(
        role="garbage",
        judge='{"passed": true, "reason": ""}',
        polish='{"reply": "x", "selected_candidate_index": 0}',
    )
    await run_pipeline(session, "x", config, llm)
    assert len(session.pipeline_trace_records) == 1
    t = session.pipeline_trace_records[0]
    assert t["final_selected_candidate_index"] == -1


async def test_pipeline_trace_written_on_polish_fallback_path() -> None:
    session = _make_session()
    config = _make_config()
    llm = _ScriptedLLM(
        role='{"reply": "ok", "goal_achieved": false, "goal_type": "", "extracted": {}}',
        judge='{"passed": true, "reason": "ok"}',
        polish="garbage",
    )
    await run_pipeline(session, "x", config, llm)
    assert len(session.pipeline_trace_records) == 1
    t = session.pipeline_trace_records[0]
    assert t["polish_output"] == "garbage"
    assert t["final_selected_candidate_index"] in {0, 1}


# ---- timeouts --------------------------------------------------------------


async def test_pipeline_role_timeout_drops_candidate() -> None:
    class _SlowLLM(LLMProvider):
        async def chat(  # type: ignore[override]
            self,
            messages: list[Message],
            *,
            json_mode: bool = False,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            await asyncio.sleep(10)
            raise AssertionError("should have timed out")

    session = _make_session()
    config = _make_config(n_roles=1)
    outcome = await run_pipeline(
        session, "x", config, _SlowLLM(), pipeline_timeout_ms=50
    )
    assert outcome.source == "default_reply"


# ---- wrap-up simplified pipeline ------------------------------------------


async def test_pipeline_wrap_up_skips_judges_and_uses_first_role() -> None:
    session = _make_session()
    config = _make_config(n_roles=3, n_judges=2)
    llm = _ScriptedLLM(
        role='{"reply": "wrap reply", "goal_achieved": false, "goal_type": "", "extracted": {}}',
        judge='{"passed": false, "reason": "should never run"}',
        polish='{"reply": "好的，wrap reply", "selected_candidate_index": 0}',
    )
    outcome = await run_pipeline(session, "x", config, llm, is_wrap_up=True)
    assert outcome.source == "wrap_up_simplified"
    # Only 1 role candidate (sort_order minimum) was used.
    assert len(session.pipeline_trace_records[0]["role_candidates"]) == 1
    # No judges were called.
    assert session.pipeline_trace_records[0]["judge_results"] == []


# ---- greeting --------------------------------------------------------------


async def test_greeting_fixed_template_skips_llm() -> None:
    session = _make_session()
    config = _make_config()
    text = await generate_greeting(
        session, config, KeywordDrivenMockLLM(), fixed_template="您好我是 AI 助手。"
    )
    assert text == "您好我是 AI 助手。"


async def test_greeting_llm_path_returns_parsed_reply() -> None:
    session = _make_session()
    config = _make_config()
    text = await generate_greeting(session, config, KeywordDrivenMockLLM())
    assert isinstance(text, str)
    assert len(text) > 0


async def test_greeting_llm_failure_returns_default() -> None:
    class _Boom(LLMProvider):
        async def chat(  # type: ignore[override]
            self,
            messages: list[Message],
            *,
            json_mode: bool = False,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            raise RuntimeError("provider down")

    session = _make_session()
    config = _make_config()
    text = await generate_greeting(session, config, _Boom())
    assert text == "您好。"


# ---- build_judge_messages (engine-judge-dialog-context change) ------------
#
# Spec: role-prompt § Requirement: Judge 拿到对话上下文.
# Covers: N 轮历史拼接 / 空历史显式占位 / SUFFIX 不在 user message 重复 /
# 用户·AI 中文前缀 + 全角冒号 / 无尾部 AI: 提示行.


def _make_judge_spec() -> JudgeSpec:
    return JudgeSpec(
        role_config_id=3,
        prompt_version_id=3,
        system_prompt="你是一个专业的对话应答分析专家...",  # PG-stored 原文形态
    )


class TestBuildJudgeMessages:
    def test_n_turn_history_rendered_in_user_message(self) -> None:
        session = _make_session()
        session.append_event("greeting", text="您好，我是智联招聘的小雨。")
        session.append_event("user_speech", text="哎你好。")
        judge = _make_judge_spec()

        messages = build_judge_messages(judge, "请问您方便聊两句吗？", session)

        assert len(messages) == 2
        user_content = messages[1].content
        # 历史段两行 + section 顺序 + 候选回复段
        assert "### 对话历史" in user_content
        assert "AI：您好，我是智联招聘的小雨。" in user_content
        assert "用户：哎你好。" in user_content
        assert "### 销售 AI 准备发给客户的候选回复" in user_content
        assert "请问您方便聊两句吗？" in user_content
        # JSON guard 兜底字面 (dashscope OpenAI-compat JSON mode 需要)
        assert user_content.endswith("按上述系统提示的 JSON schema 输出。")

    def test_empty_history_uses_explicit_placeholder(self) -> None:
        session = _make_session()
        # session.dialog_history 初始为空 — greeting 之前的纯空状态
        assert session.dialog_history == []
        judge = _make_judge_spec()

        messages = build_judge_messages(judge, "candidate text", session)

        user_content = messages[1].content
        assert "### 对话历史" in user_content
        assert "（尚无对话历史，这是首轮回复）" in user_content
        # 不留空段
        assert "### 对话历史\n\n###" not in user_content
        assert "candidate text" in user_content

    def test_schema_suffix_on_system_not_repeated_in_user(self) -> None:
        session = _make_session()
        session.append_event("greeting", text="您好。")
        judge = _make_judge_spec()

        messages = build_judge_messages(judge, "candidate", session)

        system_content = messages[0].content
        user_content = messages[1].content
        # system 含 [judge] 前缀 + PG 原文 + SUFFIX
        assert system_content.startswith("[judge] ")
        assert judge.system_prompt in system_content
        assert JUDGE_OUTPUT_SCHEMA_SUFFIX in system_content
        # user message 不重复 SUFFIX 字面
        assert JUDGE_OUTPUT_SCHEMA_SUFFIX not in user_content

    def test_user_ai_prefixes_use_chinese_fullwidth_colon(self) -> None:
        session = _make_session()
        session.append_event("greeting", text="开场白")
        session.append_event("user_speech", text="客户发言")
        judge = _make_judge_spec()

        messages = build_judge_messages(judge, "candidate", session)
        user_content = messages[1].content

        # 中文角色标签 + 全角冒号
        assert "AI：开场白" in user_content
        assert "用户：客户发言" in user_content
        # 不应出现英文标签 / 半角冒号
        assert "user:" not in user_content
        assert "assistant:" not in user_content
        assert "AI: 开场白" not in user_content  # 半角冒号 + 空格

    def test_no_trailing_ai_prompt_line(self) -> None:
        """Judge is not the speaker — must not get a trailing `AI:` cue line
        (that's role's behavior in _render_dialog)."""
        session = _make_session()
        session.append_event("greeting", text="开场白")
        session.append_event("user_speech", text="客户发言")
        judge = _make_judge_spec()

        messages = build_judge_messages(judge, "candidate text", session)
        user_content = messages[1].content

        # 历史段最后一条 dialog turn 是 "用户：客户发言"; 之后直接是空行 +
        # 候选回复段, 不应出现独占一行的 "AI：" / "AI:"
        history_block = user_content.split("### 销售 AI 准备")[0]
        # 不应有以独占行结尾的 "AI：" 或 "AI:"
        assert not history_block.rstrip().endswith("AI：")
        assert not history_block.rstrip().endswith("AI:")
        # 最后非空行应是"用户：客户发言"
        assert history_block.rstrip().splitlines()[-1] == "用户：客户发言"


