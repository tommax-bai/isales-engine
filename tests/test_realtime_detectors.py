"""Tests for interruption / silence / no-progress detectors."""

from __future__ import annotations

from isales_engine.realtime.interruption_detector import (
    InterruptionConfig,
    evaluate_partial,
)
from isales_engine.realtime.interruption_rules import default_rule
from isales_engine.realtime.no_progress_timer import is_no_progress_exceeded
from isales_engine.realtime.silence_detector import (
    SilenceConfig,
    evaluate_silence,
    silence_calc_origin_ts,
)

# ---- interruption ---------------------------------------------------------


_WHITELIST = ["嗯", "嗯嗯", "好", "好的", "哦"]


def _cfg(min_duration_ms: int = 800) -> InterruptionConfig:
    # The default tree reproduces the legacy whitelist(exact) + length(≥2) +
    # duration sequence (engine-interruption-rule-tree D4).
    return InterruptionConfig(
        whitelist=tuple(_WHITELIST),
        rule=default_rule(whitelist=_WHITELIST, min_duration_ms=min_duration_ms),
    )


def test_whitelist_keeps_speaking() -> None:
    v = evaluate_partial(
        text="嗯", speech_started_ts_ms=0, now_ts_ms=2000, config=_cfg()
    )
    assert v.verdict == "ignored"
    assert v.reason == "not(keyword)"


def test_whitelist_with_whitespace_normalised() -> None:
    v = evaluate_partial(
        text="  好的  ", speech_started_ts_ms=0, now_ts_ms=5000, config=_cfg()
    )
    assert v.verdict == "ignored"


def test_below_threshold_ignored() -> None:
    v = evaluate_partial(
        text="你看这个", speech_started_ts_ms=1000, now_ts_ms=1500, config=_cfg()
    )
    assert v.verdict == "ignored"
    assert v.reason == "duration"


def test_above_threshold_triggers() -> None:
    v = evaluate_partial(
        text="你看这个内容不错", speech_started_ts_ms=1000, now_ts_ms=2000, config=_cfg()
    )
    assert v.verdict == "triggered"


def test_min_duration_zero_treats_any_non_whitelist_as_trigger() -> None:
    cfg = InterruptionConfig(
        whitelist=("好",), rule=default_rule(whitelist=["好"], min_duration_ms=0)
    )
    assert (
        evaluate_partial(
            text="不一样", speech_started_ts_ms=0, now_ts_ms=0, config=cfg
        ).verdict
        == "triggered"
    )


def test_single_char_non_whitelist_ignored_by_length() -> None:
    # length≥2 gate: a 1-char non-whitelist partial does not interrupt.
    v = evaluate_partial(
        text="行", speech_started_ts_ms=0, now_ts_ms=5000, config=_cfg()
    )
    assert v.verdict == "ignored"
    assert v.reason == "length"


# ---- silence --------------------------------------------------------------


def test_silence_origin_uses_max_of_speech_and_tts() -> None:
    assert silence_calc_origin_ts(last_user_speech_end_ms=1000, last_tts_end_ms=2000) == 2000
    assert silence_calc_origin_ts(last_user_speech_end_ms=3000, last_tts_end_ms=2000) == 3000
    assert silence_calc_origin_ts(last_user_speech_end_ms=None, last_tts_end_ms=500) == 500
    assert silence_calc_origin_ts(last_user_speech_end_ms=None, last_tts_end_ms=None) == 0


def _silence_cfg() -> SilenceConfig:
    return SilenceConfig(
        threshold_ms=5000,
        max_activations=2,
        phrases=("您还在吗？", "请问能听到吗？"),
        hangup_phrase="那今天就先这样，再见。",
    )


def test_silence_below_threshold_waits() -> None:
    d = evaluate_silence(silence_elapsed_ms=1000, activations_so_far=0, config=_silence_cfg())
    assert d.decision == "wait"


def test_silence_first_and_second_activation_phrases() -> None:
    d1 = evaluate_silence(silence_elapsed_ms=5000, activations_so_far=0, config=_silence_cfg())
    assert d1.decision == "activate"
    assert d1.phrase == "您还在吗？"
    d2 = evaluate_silence(silence_elapsed_ms=5000, activations_so_far=1, config=_silence_cfg())
    assert d2.decision == "activate"
    assert d2.phrase == "请问能听到吗？"


def test_silence_phrases_fewer_than_max_reuse_last() -> None:
    cfg = SilenceConfig(
        threshold_ms=5000,
        max_activations=4,
        phrases=("一",),
        hangup_phrase="x",
    )
    for activations in range(4):
        d = evaluate_silence(
            silence_elapsed_ms=5000, activations_so_far=activations, config=cfg
        )
        assert d.decision == "activate"
        assert d.phrase == "一"


def test_silence_reaches_max_triggers_hangup() -> None:
    d = evaluate_silence(
        silence_elapsed_ms=5000, activations_so_far=2, config=_silence_cfg()
    )
    assert d.decision == "hangup"
    assert d.phrase == "那今天就先这样，再见。"


def test_silence_phrases_more_than_max_only_uses_first_n() -> None:
    cfg = SilenceConfig(
        threshold_ms=5000,
        max_activations=2,
        phrases=("a", "b", "c", "d"),
        hangup_phrase="x",
    )
    assert (
        evaluate_silence(
            silence_elapsed_ms=5000, activations_so_far=0, config=cfg
        ).phrase
        == "a"
    )
    assert (
        evaluate_silence(
            silence_elapsed_ms=5000, activations_so_far=1, config=cfg
        ).phrase
        == "b"
    )
    # max reached → hangup, "c"/"d" never used.
    assert (
        evaluate_silence(
            silence_elapsed_ms=5000, activations_so_far=2, config=cfg
        ).decision
        == "hangup"
    )


# ---- no_progress ----------------------------------------------------------


def test_no_progress_disabled_when_zero_or_none() -> None:
    assert not is_no_progress_exceeded(
        last_progress_ts_ms=0, now_ts_ms=1_000_000, max_no_progress_seconds=None
    )
    assert not is_no_progress_exceeded(
        last_progress_ts_ms=0, now_ts_ms=1_000_000, max_no_progress_seconds=0
    )


def test_no_progress_threshold() -> None:
    assert not is_no_progress_exceeded(
        last_progress_ts_ms=0, now_ts_ms=59_000, max_no_progress_seconds=60
    )
    assert is_no_progress_exceeded(
        last_progress_ts_ms=0, now_ts_ms=60_000, max_no_progress_seconds=60
    )
