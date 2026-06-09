"""Referee LLM — gating category decision parallel to the main streaming reply.

Spec: ai-pipeline § "referee LLM 二级决策"; role-prompt § "referee prompt 内容规范".
engine-multi-referee-and-restructure: N referees run in parallel, each with its
own prompt-defined category enum. engine-tools-multidialogue-gating: a referee
now returns a **bare category token** (a category word defined by the
gate-supervisor prompt) — no JSON, no confidence — so output is one token and the
pre-reply gate stays fast; the engine never interprets the category string — the
routing-rule decider does.

The referee takes the user's last utterance + the last ≤3 rounds of dialog
history. It runs on a cheap small model concurrently with main TTS playback and
never blocks the main link.

fail-open: any timeout / empty/invalid output collapses to a ``category=None``
result (``RefereeResult.fail_open``) which matches no routing rule (equivalent to
the old ``continue``). This is the single, protocol-level fallback for an
unreliable boundary — there is no second fallback layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

from isales_common.providers._models import Message
from isales_common.providers.llm import LLMProvider
from isales_common.schemas.pipeline import RefereeSpec

from isales_engine.call_session import DialogTurn
from isales_engine.streaming.types import (
    FAILOPEN_INVALID,
    RefereeResult,
)

logger = logging.getLogger(__name__)

# Number of most-recent dialog rounds (user+AI exchanges, ~2 entries each) the
# referee sees. Spec: "最近 ≤ 3 轮".
RECENT_ROUNDS = 3
_EMPTY_HISTORY_PLACEHOLDER = "（首轮对话，无历史）"

_USER_PRIMER = "请按系统提示只输出一个分类词，不要任何其他内容、标点或 JSON。"


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
            json_mode=False,
            temperature=referee_spec.temperature,
            top_p=referee_spec.top_p,
        )
        raw = resp.content
    except Exception:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "referee %s call failed; fail-open", referee_spec.label, exc_info=True
        )
        return RefereeResult.fail_open(
            label=referee_spec.label, reason=FAILOPEN_INVALID, duration_ms=duration_ms
        )

    duration_ms = int((time.monotonic() - start) * 1000)
    return _parse_referee_output(raw, label=referee_spec.label, duration_ms=duration_ms)


def _parse_referee_output(raw: str, *, label: str, duration_ms: int) -> RefereeResult:
    """Parse the referee's **bare-token** category (a category word defined by the
    gate-supervisor prompt).

    engine-tools-multidialogue-gating: the referee emits a single category word —
    no JSON, no confidence — so generation is one token and the pre-reply gate
    fits its tight budget. We take the first whitespace-delimited token,
    stripped of surrounding punctuation (case preserved — categories are
    case-sensitive); ``confidence`` is fixed to
    1.0 (the model no longer scores itself, so the decider's confidence floor is a
    no-op). Empty/invalid output fails open."""
    tokens = (raw or "").strip().split()
    category = tokens[0].strip(".,。，!！?？\"'`：:") if tokens else ""
    if not category:
        logger.warning("referee %s empty output: %r; fail-open", label, raw)
        return RefereeResult.fail_open(
            label=label, reason=FAILOPEN_INVALID, duration_ms=duration_ms, raw_output=raw
        )
    return RefereeResult(
        label=label,
        category=category,
        confidence=1.0,
        duration_ms=duration_ms,
        raw_output=raw,
    )
