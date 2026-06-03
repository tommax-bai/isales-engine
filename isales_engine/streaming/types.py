"""Shared value objects for the streaming pipeline (pipeline-stream-and-referee)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RefereeDecision = Literal["continue", "goal_achieved", "customer_decline", "transfer"]

DECISIONS: frozenset[str] = frozenset(
    {"continue", "goal_achieved", "customer_decline", "transfer"}
)

# Below this confidence the engine ignores the referee's decision and treats it
# as ``continue`` (role-prompt spec § referee prompt 内容规范 — confidence 评分).
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class RefereeResult:
    """Outcome of one referee LLM call.

    ``decision`` is always one of the four enum values; on any failure
    (timeout / invalid JSON / low confidence) it is forced to ``continue``
    (fail-open). ``raw_output`` keeps the unparsed model text for tracing.
    """

    decision: RefereeDecision
    goal_type: str | None = None
    confidence: float = 0.0
    duration_ms: int = 0
    raw_output: str | None = None

    @classmethod
    def fail_open(
        cls, *, duration_ms: int = 0, raw_output: str | None = None
    ) -> RefereeResult:
        return cls(
            decision="continue",
            goal_type=None,
            confidence=0.0,
            duration_ms=duration_ms,
            raw_output=raw_output,
        )
