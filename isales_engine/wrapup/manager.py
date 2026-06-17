"""Wrap-up helpers — dual counter + closing phrase.

Spec: goal-achievement § Requirement: 收尾双计数器与主动挂断;
      § Requirement: 收尾期间的特殊情况处理.

Wrap-up is an engine-INTERNAL phase (the ``session.in_wrap_up`` flag), not a
``CallStatus`` — the 4-state collapse dropped WRAPPING_UP; the observable status
stays ``IN_CALL`` through wrap-up. This module owns the round / time counter
logic + the closing-phrase pick; the gate-first turn
(``run_loop._gated_wrap_up_turn``) drives the simplified wrap-up pipeline.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class WrapUpConfig:
    max_rounds: int
    max_seconds: int
    closing_phrases: tuple[str, ...]
    # engine-wrap-up-bypass-referee: opt-in. When True the wrap-up turn runs a
    # bypass (non-gating) done-detection referee. Default False keeps wrap-up
    # referee-less (Slice-1 behavior).
    referee_enabled: bool = False


@dataclass
class WrapUpDecision:
    """``proceed=True`` → keep going (run another simplified PROCESSING);
    ``proceed=False`` → play ``closing_phrase`` and END."""

    proceed: bool
    closing_phrase: str
    # On proceed=False this value is persisted verbatim as the
    # ``wrap_up_completed`` transcript event's ``reason`` (run_loop.py), so it
    # MUST stay within that event's spec vocabulary
    # ``Literal["max_rounds", "max_seconds"]`` (transcript spec § event table).
    # "ok" is the proceed=True sentinel and is never persisted.
    reason: str  # "max_rounds" / "max_seconds" / "ok"


def evaluate_wrap_up(
    *,
    rounds_so_far: int,
    started_at_monotonic: float | None,
    config: WrapUpConfig,
    now_monotonic: float | None = None,
) -> WrapUpDecision:
    """Decide whether the next user turn should still loop in wrap-up."""

    if rounds_so_far >= config.max_rounds:
        return WrapUpDecision(
            proceed=False,
            closing_phrase=_pick_phrase(config),
            reason="max_rounds",
        )
    if started_at_monotonic is not None:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        if now - started_at_monotonic >= config.max_seconds:
            return WrapUpDecision(
                proceed=False,
                closing_phrase=_pick_phrase(config),
                reason="max_seconds",
            )
    return WrapUpDecision(proceed=True, closing_phrase="", reason="ok")


def _pick_phrase(config: WrapUpConfig) -> str:
    if not config.closing_phrases:
        return "好的，再见。"
    return random.choice(config.closing_phrases)
