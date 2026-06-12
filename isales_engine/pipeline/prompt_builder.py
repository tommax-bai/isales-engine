"""Assemble system / user messages for the main LLM call.

Spec: role-prompt § "System message 内容" / "User message 拼接结构" /
"收尾期间在 system prompt 末尾追加指令" / "跟进通话的 prompt 增强".

pipeline-stream-and-referee: the judge / polish builders and the
ROLE/JUDGE/POLISH_OUTPUT_SCHEMA_SUFFIX constants are gone. The main LLM now
emits **plain text** (no JSON Mode), so the engine MUST NOT inject any
output-format suffix — the campaign-authored prompt owns its output rules
(role-prompt § "输出格式约束必须保留" — engine MUST NOT 强制注入约束). The only
engine-appended segment is the WRAPPING_UP closing instruction, which MUST be
last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isales_common.providers._models import Message
from isales_common.schemas.pipeline import (
    ExtractorSpec,
    MainSpec,
    PersonaSpec,
    RefereeSpec,
    RestructureSpec,
)

from isales_engine.call_session import CallSession

# Appended to the main system prompt, last, during WRAPPING_UP only
# (role-prompt § 收尾期间在 system prompt 末尾追加指令).
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
class LeadInfo:
    name: str | None
    phone: str
    custom_data: dict[str, Any]


@dataclass
class PipelineConfig:
    """Engine-side runtime pipeline config.

    Composes the three shared LLM slots (``isales_common.schemas.pipeline``)
    with the per-call rendering inputs the prompt builder needs. The common
    ``PipelineConfig`` is the minimal data contract; this is the richer runtime
    object the engine actually threads through run_loop / orchestrator.
    """

    main: MainSpec
    # engine-multi-referee-and-restructure: N parallel referees (was a single
    # ``referee``) + an optional restructure (re-voice) slot + the campaign's
    # ordered routing rules and restructure caps.
    referees: list[RefereeSpec]
    extractor: ExtractorSpec
    default_replies: list[str]
    lead: LeadInfo
    restructure: RestructureSpec | None = None
    # Raw routing rules (list of dicts) as stored in campaign.routing_rules.
    routing_rules: list[dict[str, Any]] = field(default_factory=list)
    max_continuous_restructure: int = 2
    # engine-auto-restructure-on-interrupt: when True, a barge-in-interrupted turn
    # (session.interrupt_remaining_text set) with no explicit routing-rule match
    # falls through to restructure(interrupt_remaining) instead of continue. The
    # referee gate + explicit rules stay the veto (first-match-wins).
    auto_restructure_on_interrupt: bool = False
    last_call_summary: str | None = None
    follow_up_count: int = 0
    short_reply_active: bool = False
    # Snapshot helper: True once a WRAPPING_UP turn appended the closing
    # instruction (written to call_record.prompt_versions.wrap_up_appended).
    wrap_up_appended: bool = field(default=False)
    # gating + multi-persona (engine-tools-multidialogue-gating). ``personas``:
    # opt-in speculative dialogue slots run eagerly in parallel with main; the
    # referee gate selects one and cancels the rest. ``tools``: raw alias →
    # hangup/transfer config dicts (campaign.tools JSONB, stored raw like
    # ``routing_rules`` — the tool route reads ``closing_phrase`` / ``interrupt``
    # off the dict). ``persona_fanout_cap``: total speculative routes incl main,
    # clamped to [1,3] by the engine (default 1 = main only, no speculation).
    # ``referee_timeout_ms`` / ``referee_fail_open_route``: pre-reply gating
    # timeout + the dialogue route released when the gate times out / fails open.
    personas: list[PersonaSpec] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)
    persona_fanout_cap: int = 1
    referee_timeout_ms: int = 600
    referee_fail_open_route: str = "main"


def build_main_messages(
    session: CallSession,
    config: PipelineConfig,
    *,
    is_wrap_up: bool,
) -> list[Message]:
    """Assemble the [system, user] messages for one main LLM streaming call.

    The whole conversation goes in a single user message (role-prompt § "不使用
    标准 chat multi-turn"). No output-format suffix is injected.
    """
    system = config.main.system_prompt
    if config.follow_up_count > 0:
        system += FOLLOW_UP_APPEND_TEMPLATE.format(n=config.follow_up_count)
    if config.short_reply_active:
        system += SHORT_REPLY_APPEND
    # WRAP_UP_APPEND MUST be last (role-prompt § WRAPPING_UP 进入即追加).
    if is_wrap_up:
        system += WRAP_UP_APPEND

    user = _build_user_message(session, config)
    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_greeting_messages(config: PipelineConfig) -> list[Message]:
    """Assemble messages for the LLM-generated greeting (main slot, plain text)."""
    system = config.main.system_prompt
    user = (
        f"【线索信息】name={config.lead.name or '—'}, phone={config.lead.phone}\n"
        "【任务】请生成开场白，直接输出要对客户说的话。"
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
