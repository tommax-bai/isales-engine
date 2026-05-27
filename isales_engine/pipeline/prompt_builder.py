"""Assemble system / user messages for role / judge / polish LLM calls.

Spec: role-prompt § Requirement: Prompt 三段式组装; § 收尾期间在 system prompt
末尾追加指令; § 跟进通话的 prompt 增强.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isales_common.providers._models import Message

from isales_engine.call_session import CallSession

WRAP_UP_APPEND = """

---
【当前状态：收尾对话】
目标已达成。请简短确认或告别后结束对话，不要再尝试推进新议题。
"""

FOLLOW_UP_APPEND_TEMPLATE = """

【跟进上下文】
这是对该用户的第 {n} 次跟进。请根据下面的【上次通话纪要】调整开场和后续话术，避免重复内容。
"""

SHORT_REPLY_APPEND = "\n\n【打断保护】请用一句话回应。"

# Output-schema suffix appended to the SYSTEM prompt of role/judge/polish.
# Necessary because the PG-stored system prompts may use legacy phrasing
# like "直接返回 POSITIVE/NEGATIVE" or "不要输出任何额外的解释,标识等等",
# which conflict with the parser's expected JSON schema. The suffix is
# explicitly framed as an OVERRIDE so the model treats it as authoritative
# over any earlier 输出规则 instruction in the same system message.
# (dashscope OpenAI-compat JSON mode also requires the substring 'json' to
# appear somewhere in messages; this suffix satisfies that as a side-effect.)
ROLE_OUTPUT_SCHEMA_SUFFIX = (
    "\n\n【输出规则·最高优先级】\n"
    "无论上述任何输出规则如何描述,你的最终输出必须且只能是一个 JSON 对象,\n"
    "schema 如下,字段名一字不差,缺一不可:\n"
    '{"reply": "<面向客户的回复文本>",'
    ' "goal_achieved": <true|false>,'
    ' "goal_type": "<目标类型字符串,无则空串>",'
    ' "extracted": {<结构化字段对象,无则空对象>}}\n'
    "不要输出 markdown 代码块围栏,不要输出任何解释/标识/前后缀文本。"
)
JUDGE_OUTPUT_SCHEMA_SUFFIX = (
    "\n\n【输出规则·最高优先级】\n"
    "无论上述任何输出规则如何描述,你的最终输出必须且只能是一个 JSON 对象,\n"
    "schema 如下,字段名一字不差,缺一不可:\n"
    '{"passed": <true|false>, "reason": "<简要原因>"}\n'
    "判断为接受/POSITIVE 时 passed=true; 判断为拒绝/NEGATIVE 时 passed=false。\n"
    "不要输出 markdown 代码块围栏,不要输出任何解释/标识/前后缀文本。"
)
POLISH_OUTPUT_SCHEMA_SUFFIX = (
    "\n\n【输出规则·最高优先级】\n"
    "无论上述任何输出规则如何描述,你的最终输出必须且只能是一个 JSON 对象,\n"
    "schema 如下,字段名一字不差,缺一不可:\n"
    '{"reply": "<润色后的最终回复文本>",'
    ' "selected_candidate_index": <从 0 开始的整数,标识被选中的候选索引>}\n'
    "不要输出 markdown 代码块围栏,不要输出任何解释/标识/前后缀文本。"
)


@dataclass
class RoleSpec:
    role_config_id: int
    prompt_version_id: int
    system_prompt: str
    model: str = "mock"
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class JudgeSpec:
    role_config_id: int
    prompt_version_id: int
    system_prompt: str
    model: str = "mock"
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class PolishSpec:
    role_config_id: int
    prompt_version_id: int
    system_prompt: str
    model: str = "mock"
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class LeadInfo:
    name: str | None
    phone: str
    custom_data: dict[str, Any]


@dataclass
class PipelineConfig:
    roles: list[RoleSpec]
    judges: list[JudgeSpec]
    polish: PolishSpec
    default_replies: list[str]
    lead: LeadInfo
    last_call_summary: str | None = None
    follow_up_count: int = 0
    short_reply_active: bool = False


def build_role_messages(
    session: CallSession,
    role: RoleSpec,
    config: PipelineConfig,
    *,
    is_wrap_up: bool,
) -> list[Message]:
    system = "[role] " + role.system_prompt
    if is_wrap_up:
        system += WRAP_UP_APPEND
    if config.follow_up_count > 0:
        system += FOLLOW_UP_APPEND_TEMPLATE.format(n=config.follow_up_count)
    if config.short_reply_active:
        system += SHORT_REPLY_APPEND
    system += ROLE_OUTPUT_SCHEMA_SUFFIX

    user = _build_user_message(session, config)
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_judge_messages(judge: JudgeSpec, candidate_reply: str) -> list[Message]:
    system = "[judge] " + judge.system_prompt + JUDGE_OUTPUT_SCHEMA_SUFFIX
    user = (
        f"请审查以下候选回复：\n{candidate_reply}\n\n"
        "按上述系统提示的 JSON schema 输出。"
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_polish_messages(
    polish: PolishSpec, candidates: list[str]
) -> list[Message]:
    system = "[polish] " + polish.system_prompt + POLISH_OUTPUT_SCHEMA_SUFFIX
    body_lines = [f"candidate[{i}]: {reply}" for i, reply in enumerate(candidates)]
    user = (
        "请润色并选优：\n"
        + "\n".join(body_lines)
        + "\n\n按上述系统提示的 JSON schema 输出。"
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_greeting_messages(role: RoleSpec, config: PipelineConfig) -> list[Message]:
    system = "[role] " + role.system_prompt + ROLE_OUTPUT_SCHEMA_SUFFIX
    user = (
        f"【线索信息】name={config.lead.name or '—'}, phone={config.lead.phone}\n"
        "【任务】请生成开场白。\n"
        "按上述系统提示的 JSON schema 输出。"
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _build_user_message(session: CallSession, config: PipelineConfig) -> str:
    parts: list[str] = []
    if config.last_call_summary and config.follow_up_count > 0:
        parts.append("【上次通话纪要】\n" + config.last_call_summary)
    parts.append(_render_lead_info(config.lead))
    parts.append(_render_dialog(session))
    # Schema requirement lives in ROLE_OUTPUT_SCHEMA_SUFFIX on the system prompt,
    # not here — system has higher precedence and the PG-stored role prompt
    # otherwise dominates. Keeping the literal substring "JSON" here for the
    # dashscope OpenAI-compat 'messages must contain json' guard.
    parts.append("按系统提示的 JSON schema 输出。")
    return "\n\n".join(parts)


def _render_lead_info(lead: LeadInfo) -> str:
    custom = ", ".join(f"{k}={v}" for k, v in lead.custom_data.items()) or "—"
    return f"【线索信息】name={lead.name or '—'}, phone={lead.phone}, custom_data={custom}"


def _render_dialog(session: CallSession) -> str:
    lines = ["【对话】"]
    for turn in session.dialog_history:
        prefix = "用户" if turn.role == "user" else "AI"
        lines.append(f"{prefix}: {turn.text}")
    lines.append("AI:")
    return "\n".join(lines)
