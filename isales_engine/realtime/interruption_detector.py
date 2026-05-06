"""Double-condition interruption detection.

Spec: interruption-detection § all Requirements.

The detector is fed ASR partial results during ``SPEAKING`` / ``FILLER`` and
emits an ``InterruptionVerdict`` per partial:

* ``ignored`` — at least one bypass condition matched (whitelist OR duration
  below threshold).
* ``triggered`` — both conditions failed; the caller MUST stop TTS / filler
  and treat the eventual ASR final as a new PROCESSING input.

Once ``triggered`` fires, subsequent partials remain ``triggered`` (verdict
is unconditional after first hit) per spec § "打断判定不可撤销".

Connected counters (consecutive interruption + reset on full SPEAKING) are
maintained on :class:`CallSession`; the detector is a stateless function plus
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["ignored", "triggered"]


@dataclass
class InterruptionConfig:
    whitelist: tuple[str, ...]
    min_duration_ms: int


@dataclass
class InterruptionVerdict:
    verdict: Verdict
    reason: str  # "whitelist" / "below_threshold" / "exceeds"


def evaluate_partial(
    *, text: str, speech_started_ts_ms: int, now_ts_ms: int, config: InterruptionConfig
) -> InterruptionVerdict:
    """Apply double-condition rule. Inputs are pre-normalised by the caller."""

    if text.strip() in config.whitelist:
        return InterruptionVerdict(verdict="ignored", reason="whitelist")

    elapsed = now_ts_ms - speech_started_ts_ms
    if elapsed < config.min_duration_ms:
        return InterruptionVerdict(verdict="ignored", reason="below_threshold")

    return InterruptionVerdict(verdict="triggered", reason="exceeds")
