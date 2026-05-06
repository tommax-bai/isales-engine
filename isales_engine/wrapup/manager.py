"""WRAPPING_UP state helpers — dual counter + closing phrase.

Spec: goal-achievement § Requirement: 收尾双计数器与主动挂断;
      § Requirement: 收尾期间的特殊情况处理.

The state machine transition into WRAPPING_UP and the simplified pipeline
itself live elsewhere (PR #11 + PR #6's ``run_pipeline(is_wrap_up=True)``).
This module owns the round / time counter logic and the closing-phrase pick.
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


@dataclass
class WrapUpDecision:
    """``proceed=True`` → keep going (run another simplified PROCESSING);
    ``proceed=False`` → play ``closing_phrase`` and END."""

    proceed: bool
    closing_phrase: str
    reason: str  # "rounds_exhausted" / "seconds_exhausted" / "ok"


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
            reason="rounds_exhausted",
        )
    if started_at_monotonic is not None:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        if now - started_at_monotonic >= config.max_seconds:
            return WrapUpDecision(
                proceed=False,
                closing_phrase=_pick_phrase(config),
                reason="seconds_exhausted",
            )
    return WrapUpDecision(proceed=True, closing_phrase="", reason="ok")


def _pick_phrase(config: WrapUpConfig) -> str:
    if not config.closing_phrases:
        return "好的，再见。"
    return random.choice(config.closing_phrases)
