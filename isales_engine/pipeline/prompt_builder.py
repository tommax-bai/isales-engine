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

    user = _build_user_message(session, config)
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_judge_messages(judge: JudgeSpec, candidate_reply: str) -> list[Message]:
    system = "[judge] " + judge.system_prompt
    user = f"请审查以下候选回复：\n{candidate_reply}"
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_polish_messages(
    polish: PolishSpec, candidates: list[str]
) -> list[Message]:
    system = "[polish] " + polish.system_prompt
    body_lines = [f"candidate[{i}]: {reply}" for i, reply in enumerate(candidates)]
    user = "请润色并选优：\n" + "\n".join(body_lines)
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_greeting_messages(role: RoleSpec, config: PipelineConfig) -> list[Message]:
    system = "[role] " + role.system_prompt
    user = (
        f"【线索信息】name={config.lead.name or '—'}, phone={config.lead.phone}\n"
        "【任务】请生成开场白。\n"
        "请以 JSON 格式回复,例如 {\"reply\": \"开场白文本\"}。"
        " 仅输出 JSON,不要其他说明。"
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _build_user_message(session: CallSession, config: PipelineConfig) -> str:
    parts: list[str] = []
    if config.last_call_summary and config.follow_up_count > 0:
        parts.append("【上次通话纪要】\n" + config.last_call_summary)
    parts.append(_render_lead_info(config.lead))
    parts.append(_render_dialog(session))
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
