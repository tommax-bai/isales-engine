"""Composable rule-tree interruption detection.

Spec: interruption-detection § "可组合规则树打断判定" + "打断规则树配置与装配".
Change: engine-interruption-rule-tree (replaces the old whitelist→length→duration
fixed sequence with a configurable rule tree; port of voxen core/interruption/).

The detector is fed ASR partial results during ``SPEAKING`` / ``FILLER`` and emits
an ``InterruptionVerdict`` per partial:

* ``ignored`` — the rule tree evaluated to false; the caller keeps playing TTS and
  discards the user input.
* ``triggered`` — the rule tree evaluated to true; the caller MUST stop TTS / filler
  and treat the eventual ASR final as a new PROCESSING input.

Once ``triggered`` fires, subsequent partials remain ``triggered`` (verdict is
unconditional after first hit) per spec § "打断判定不可撤销" — enforced by the
caller (``_partial_monitor`` latches ``session.interruption_signaled``).

The rule tree is compiled ONCE when ``InterruptionConfig`` is assembled (see
``runtime_config``); ``evaluate_partial`` only evaluates it per partial.
``whitelist`` is retained for a DIFFERENT consumer — the no-progress filter in
``_await_user_or_silence`` (a final that is purely a filler phrase advances
nothing) — it is NOT the barge-in decision anymore (the rule tree is).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from isales_engine.realtime.interruption_rules import Rule, RuleInput

Verdict = Literal["ignored", "triggered"]


@dataclass
class InterruptionConfig:
    whitelist: tuple[str, ...]  # no-progress filter in _await_user_or_silence
    rule: Rule  # compiled barge-in rule tree (the barge-in decision)


@dataclass
class InterruptionVerdict:
    verdict: Verdict
    reason: str  # provenance of the deciding rule leaf (e.g. "keyword"/"length"/"and")


def evaluate_partial(
    *, text: str, speech_started_ts_ms: int, now_ts_ms: int, config: InterruptionConfig
) -> InterruptionVerdict:
    """Run the campaign's barge-in rule tree on one ASR partial.

    Inputs are pre-normalised by the caller. The rule tree decides; leaves test
    the partial text and the elapsed speech duration (now - speech_start).
    """
    elapsed_ms = now_ts_ms - speech_started_ts_ms
    verdict = config.rule.evaluate(RuleInput(text=text, elapsed_ms=elapsed_ms))
    if verdict.interrupt:
        return InterruptionVerdict(verdict="triggered", reason=verdict.reason)
    return InterruptionVerdict(verdict="ignored", reason=verdict.reason)
