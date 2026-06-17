"""Silence threshold detector + activation phrase selection.

Spec: silence-activation § all Requirements.

The detector is *passive* — its API is small enough that the LISTENING-state
loop in PR #11 owns the timer; this module just answers "given that the user
has been silent for N ms, what should happen?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["wait", "activate", "hangup"]


@dataclass
class SilenceConfig:
    threshold_ms: int
    max_activations: int
    phrases: tuple[str, ...]
    hangup_phrase: str
    # engine-wrap-up-silence-hangup: WRAPPING_UP-phase silence hangup window (ms).
    # Lives here (not on WrapUpConfig) because _await_user_or_silence only receives
    # the SilenceConfig. Built from campaign.wrap_up_silence_hangup_ms.
    wrap_up_hangup_ms: int = 6000


@dataclass
class SilenceDecision:
    decision: Decision
    phrase: str | None  # text to TTS for activate / hangup


def silence_calc_origin_ts(
    *, last_user_speech_end_ms: int | None, last_tts_end_ms: int | None
) -> int:
    """Compute the silence-counter origin per spec § "沉默检测与激活触发"."""

    if last_user_speech_end_ms is None and last_tts_end_ms is None:
        return 0
    return max(last_user_speech_end_ms or 0, last_tts_end_ms or 0)


def evaluate_silence(
    *,
    silence_elapsed_ms: int,
    activations_so_far: int,
    config: SilenceConfig,
    in_wrap_up: bool = False,
) -> SilenceDecision:
    # engine-wrap-up-silence-hangup: in WRAPPING_UP the customer has already heard
    # the farewell, so going silent is a terminal signal — hang up directly after
    # wrap_up_hangup_ms with NO phrase, and NEVER run the "你好，还在么？"
    # reactivation ladder (that stays main-phase only). This is the call-215 fix.
    # Removal trigger: drop this branch if the Slice-2 wrap-up bypass referee ever
    # fully owns wrap-up termination.
    if in_wrap_up:
        if silence_elapsed_ms >= config.wrap_up_hangup_ms:
            return SilenceDecision(decision="hangup", phrase=None)
        return SilenceDecision(decision="wait", phrase=None)

    if silence_elapsed_ms < config.threshold_ms:
        return SilenceDecision(decision="wait", phrase=None)

    if activations_so_far >= config.max_activations:
        return SilenceDecision(decision="hangup", phrase=config.hangup_phrase)

    # Pick phrase by index, with last-element reuse when phrases < max.
    if not config.phrases:
        # No phrases configured → spec § "话术数小于激活上限" trivially applies.
        return SilenceDecision(decision="activate", phrase="")
    idx = min(activations_so_far, len(config.phrases) - 1)
    return SilenceDecision(decision="activate", phrase=config.phrases[idx])
