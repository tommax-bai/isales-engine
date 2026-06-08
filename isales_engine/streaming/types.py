"""Shared value objects for the streaming pipeline.

engine-multi-referee-and-restructure: a referee no longer emits a strong
``decision`` enum. Each referee emits a free ``{category, confidence}`` whose
semantics live in that referee's own prompt; the engine never interprets the
category string — it hands ``{label: category}`` to the routing-rule decider.
"""

from __future__ import annotations

from dataclasses import dataclass

# Below this confidence a referee's category is treated as "no usable output"
# for rule matching (role-prompt spec § confidence 评分): such a referee simply
# matches no rule and the turn falls through to continue (general confidence
# floor — independent of any restructure trigger).
CONFIDENCE_THRESHOLD = 0.7

# Fail-open markers recorded in pipeline_trace's referee category when a referee
# produced no usable category (vs. a real prompt-defined enum value).
FAILOPEN_TIMEOUT = "timeout"
FAILOPEN_INVALID = "invalid"


@dataclass
class RefereeResult:
    """Outcome of one referee LLM call.

    ``category`` is the raw prompt-defined classification string, or ``None``
    when the call failed / output was invalid (fail-open). ``failopen_reason``
    records *why* it is None for tracing. ``raw_output`` keeps the unparsed
    model text.
    """

    label: str
    category: str | None = None
    confidence: float = 0.0
    duration_ms: int = 0
    raw_output: str | None = None
    # "timeout" / "invalid" when category is None.
    failopen_reason: str | None = None

    @classmethod
    def fail_open(
        cls,
        *,
        label: str,
        reason: str = FAILOPEN_INVALID,
        duration_ms: int = 0,
        raw_output: str | None = None,
    ) -> RefereeResult:
        return cls(
            label=label,
            category=None,
            confidence=0.0,
            duration_ms=duration_ms,
            raw_output=raw_output,
            failopen_reason=reason,
        )

    def effective_category(self, threshold: float = CONFIDENCE_THRESHOLD) -> str | None:
        """Category used for rule matching: ``None`` when not usable (fail-open
        or below the confidence floor)."""
        if self.category is None:
            return None
        if self.confidence < threshold:
            return None
        return self.category

    def trace_category(self, threshold: float = CONFIDENCE_THRESHOLD) -> str:
        """The category string recorded in pipeline_trace (real enum or marker)."""
        if self.category is None:
            return self.failopen_reason or FAILOPEN_INVALID
        return self.category

    def as_trace(self, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
        """One ``referee_results`` array element (transcript spec)."""
        return {
            "label": self.label,
            "category": self.trace_category(threshold),
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
        }
