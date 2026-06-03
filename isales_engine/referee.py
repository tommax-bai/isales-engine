"""Referee LLM — side-band enum decision parallel to the main streaming reply.

Spec: ai-pipeline § "referee 二级决策"; role-prompt § "referee prompt 内容规范";
goal-achievement § "goal_achieved 触发来源" (referee decision drives the state
machine instead of the old polish JSON field).

The referee takes the user's last utterance + the last ≤3 rounds of dialog
history and returns ``{decision, goal_type, confidence}``. It runs on a cheap
small model concurrently with main TTS playback and never blocks the main link.

fail-open: any timeout / invalid JSON / low-confidence result collapses to
``continue`` so a referee failure never drops the call (same semantics as the
old ``_default_reply_outcome``). This is the single, protocol-level fallback for
an unreliable boundary — there is no second fallback layer.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable

from isales_common.providers._models import Message
from isales_common.providers.llm import LLMProvider
from isales_common.schemas.pipeline import RefereeSpec

from isales_engine.call_session import DialogTurn
from isales_engine.streaming.types import (
    CONFIDENCE_THRESHOLD,
    DECISIONS,
    RefereeResult,
)

logger = logging.getLogger(__name__)

# Number of most-recent dialog rounds (user+AI exchanges, ~2 entries each) the
# referee sees. Spec: "最近 ≤ 3 轮".
RECENT_ROUNDS = 3
_EMPTY_HISTORY_PLACEHOLDER = "（首轮对话，无历史）"

_USER_PRIMER = "请根据以上输入，按指定 JSON schema 输出决策。"


def _render_dialog_history_for_referee(dialog_history: Iterable[DialogTurn]) -> str:
    """Render dialog history as ``用户：xxx / AI：xxx`` lines.

    Inherits the 5/29 ``engine-judge-dialog-context`` rendering rules: 全角冒号,
    ``用户：`` / ``AI：`` prefixes, explicit placeholder for empty history.
    """
    turns = list(dialog_history)
    if not turns:
        return _EMPTY_HISTORY_PLACEHOLDER
    lines: list[str] = []
    for turn in turns:
        prefix = "用户" if turn.role == "user" else "AI"
        lines.append(f"{prefix}：{turn.text}")
    return "\n".join(lines)


def recent_dialog_rounds(
    dialog_history: list[DialogTurn], *, rounds: int = RECENT_ROUNDS
) -> list[DialogTurn]:
    """Return the most-recent ``rounds`` rounds (≈ 2 entries each) of history."""
    if not dialog_history:
        return []
    return dialog_history[-(rounds * 2) :]


async def run_referee(
    session: object,
    user_last_utterance: str,
    recent_dialog_history: list[DialogTurn],
    referee_spec: RefereeSpec,
    llm: LLMProvider,
) -> RefereeResult:
    """Run the referee LLM and return a validated :class:`RefereeResult`.

    ``referee_spec.system_prompt`` is the campaign-authored referee prompt
    containing ``{{user_last_utterance}}`` / ``{{recent_dialog_history}}``
    placeholders, which are substituted here. Returns a fail-open ``continue``
    on any error.
    """
    rendered_history = _render_dialog_history_for_referee(recent_dialog_history)
    system = referee_spec.system_prompt.replace(
        "{{user_last_utterance}}", user_last_utterance or ""
    ).replace("{{recent_dialog_history}}", rendered_history)
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=_USER_PRIMER),
    ]

    start = time.monotonic()
    try:
        resp = await llm.chat(
            messages,
            json_mode=True,
            temperature=referee_spec.temperature,
            top_p=referee_spec.top_p,
        )
        raw = resp.content
    except Exception:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.warning("referee call failed; fail-open continue", exc_info=True)
        return RefereeResult.fail_open(duration_ms=duration_ms)

    duration_ms = int((time.monotonic() - start) * 1000)
    return _parse_referee_output(raw, duration_ms=duration_ms)


def _parse_referee_output(raw: str, *, duration_ms: int) -> RefereeResult:
    """Validate the referee JSON; fail-open ``continue`` on any problem.

    Applies the confidence floor: a non-``continue`` decision with
    ``confidence < CONFIDENCE_THRESHOLD`` is downgraded to ``continue``.
    """
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        logger.warning("referee output not JSON: %r; fail-open", raw)
        return RefereeResult.fail_open(duration_ms=duration_ms, raw_output=raw)

    if not isinstance(obj, dict):
        return RefereeResult.fail_open(duration_ms=duration_ms, raw_output=raw)

    decision = obj.get("decision")
    if decision not in DECISIONS:
        logger.warning("referee decision invalid: %r; fail-open", decision)
        return RefereeResult.fail_open(duration_ms=duration_ms, raw_output=raw)

    goal_type = obj.get("goal_type")
    if goal_type is not None and not isinstance(goal_type, str):
        goal_type = None

    confidence = obj.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    # Confidence floor: low-confidence non-continue decisions are ignored.
    if decision != "continue" and confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "referee decision %s below confidence floor (%.2f); -> continue",
            decision,
            confidence,
        )
        return RefereeResult(
            decision="continue",
            goal_type=None,
            confidence=confidence,
            duration_ms=duration_ms,
            raw_output=raw,
        )

    # goal_achieved requires a goal_type; without one fall back to continue.
    if decision == "goal_achieved" and not goal_type:
        logger.warning("referee goal_achieved without goal_type; -> continue")
        return RefereeResult(
            decision="continue",
            goal_type=None,
            confidence=confidence,
            duration_ms=duration_ms,
            raw_output=raw,
        )

    return RefereeResult(
        decision=decision,
        goal_type=goal_type,
        confidence=confidence,
        duration_ms=duration_ms,
        raw_output=raw,
    )
