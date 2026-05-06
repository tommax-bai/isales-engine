"""Two-step JSON parsing for role LLM output.

Spec: role-prompt § Requirement: JSON Mode 强制策略（两步保护）;
      ai-pipeline § Requirement: 角色 LLM JSON 解析失败的处理.

Step 1: ``json.loads(text)`` direct.
Step 2: extract ``{...}`` block via regex and re-parse.
Step 3 (fallback): mark ``parse_failed=True``; the candidate is dropped by
the orchestrator (equivalent to "judge rejected").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class RoleParseResult:
    reply: str = ""
    goal_achieved: bool = False
    goal_type: str = ""
    extracted: dict[str, Any] = field(default_factory=dict)
    parse_failed: bool = False
    raw: str = ""


def parse_role_output(text: str) -> RoleParseResult:
    """Parse a role LLM's response text. Never raises — returns ``parse_failed=True``."""

    candidate = text.strip()
    parsed = _try_parse(candidate)
    if parsed is None:
        m = _BLOCK_RE.search(candidate)
        if m is not None:
            parsed = _try_parse(m.group(0))
    if parsed is None:
        return RoleParseResult(parse_failed=True, raw=text)

    reply = str(parsed.get("reply", "")).strip()
    if not reply:
        return RoleParseResult(parse_failed=True, raw=text)

    goal_achieved = bool(parsed.get("goal_achieved", False))
    goal_type = str(parsed.get("goal_type", ""))
    extracted = parsed.get("extracted", {})
    if not isinstance(extracted, dict):
        extracted = {}

    return RoleParseResult(
        reply=reply,
        goal_achieved=goal_achieved,
        goal_type=goal_type,
        extracted=extracted,
        parse_failed=False,
        raw=text,
    )


def parse_judge_output(text: str) -> tuple[bool, str]:
    """Return ``(passed, reason)``. Parse failures default to ``passed=False``
    so a malformed judge cannot accidentally let bad content through."""

    parsed = _try_parse(text.strip())
    if parsed is None:
        m = _BLOCK_RE.search(text)
        if m is not None:
            parsed = _try_parse(m.group(0))
    if parsed is None:
        return False, "judge_parse_failed"
    return bool(parsed.get("passed", False)), str(parsed.get("reason", ""))


def parse_polish_output(text: str) -> tuple[str | None, int | None]:
    """Return ``(reply, selected_candidate_index)`` or ``(None, None)`` on failure."""

    parsed = _try_parse(text.strip())
    if parsed is None:
        m = _BLOCK_RE.search(text)
        if m is not None:
            parsed = _try_parse(m.group(0))
    if parsed is None:
        return None, None
    reply = parsed.get("reply")
    idx = parsed.get("selected_candidate_index")
    if not isinstance(reply, str) or not reply.strip():
        return None, None
    if not isinstance(idx, int):
        return None, None
    return reply.strip(), idx


def _try_parse(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return None
    return result if isinstance(result, dict) else None
