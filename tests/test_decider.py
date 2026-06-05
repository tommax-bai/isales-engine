"""Tests for the routing-rule decider (engine-multi-referee-and-restructure §4.3)."""

from __future__ import annotations

from isales_engine.pipeline.decider import decide
from isales_engine.streaming.types import RefereeResult


def _r(label, category, confidence=0.9):
    return RefereeResult(label=label, category=category, confidence=confidence)


GOAL_RULE = {
    "referee": "main_judge",
    "match": ["goal_achieved"],
    "action": {"type": "transition", "to": "goal_achieved", "goal_type": "appointment"},
}
TRANSFER_RULE = {
    "referee": "reject",
    "match": ["OPERATOR"],
    "action": {"type": "transition", "to": "transfer"},
}
RESTRUCTURE_RULE = {
    "referee": "intent",
    "match": ["NEGATIVE"],
    "action": {"type": "restructure", "source": "last_reply"},
}


def test_no_referees_continue():
    action = decide([], [GOAL_RULE])
    assert action.kind == "continue"
    assert action.matched_rule is None


def test_first_match_wins():
    # Both reject→transfer and intent→restructure would match; transfer is first.
    results = [_r("reject", "OPERATOR"), _r("intent", "NEGATIVE")]
    rules = [TRANSFER_RULE, RESTRUCTURE_RULE]
    action = decide(results, rules, restructure_enabled=True)
    assert action.kind == "transition"
    assert action.to == "transfer"
    assert action.matched_rule == TRANSFER_RULE


def test_order_matters_restructure_first():
    results = [_r("reject", "OPERATOR"), _r("intent", "NEGATIVE")]
    rules = [RESTRUCTURE_RULE, TRANSFER_RULE]
    action = decide(results, rules, restructure_enabled=True)
    assert action.kind == "restructure"
    assert action.source == "last_reply"


def test_goal_achieved_carries_goal_type():
    action = decide([_r("main_judge", "goal_achieved")], [GOAL_RULE])
    assert action.kind == "transition"
    assert action.to == "goal_achieved"
    assert action.goal_type == "appointment"


def test_no_match_continue():
    action = decide([_r("main_judge", "continue")], [GOAL_RULE])
    assert action.kind == "continue"


def test_low_confidence_referee_does_not_match():
    # category matches the rule but confidence is below the floor → no match.
    action = decide([_r("main_judge", "goal_achieved", confidence=0.5)], [GOAL_RULE])
    assert action.kind == "continue"


def test_failopen_referee_does_not_match():
    action = decide([RefereeResult.fail_open(label="main_judge", reason="timeout")], [GOAL_RULE])
    assert action.kind == "continue"


def test_restructure_action_degrades_without_restructure_slot():
    results = [_r("intent", "NEGATIVE")]
    action = decide(results, [RESTRUCTURE_RULE], restructure_enabled=False)
    assert action.kind == "continue"
    assert action.matched_rule == RESTRUCTURE_RULE  # recorded for trace


def test_low_confidence_primary_triggers_restructure():
    # No rule matches, but the primary referee returned a real category below
    # the floor → low-confidence restructure (source=last_reply).
    results = [_r("main_judge", "goal_achieved", confidence=0.4)]
    action = decide(
        results,
        [GOAL_RULE],
        primary_referee_label="main_judge",
        restructure_enabled=True,
    )
    assert action.kind == "restructure"
    assert action.source == "last_reply"
    assert action.restructure_trigger == "low_confidence"


def test_low_confidence_primary_no_restructure_when_disabled():
    results = [_r("main_judge", "goal_achieved", confidence=0.4)]
    action = decide(
        results, [GOAL_RULE], primary_referee_label="main_judge", restructure_enabled=False
    )
    assert action.kind == "continue"


def test_low_confidence_fallback_needs_parseable_category():
    # A hard fail-open (category None) on the primary referee → continue, not
    # restructure (only a genuine low-confidence parse triggers re-voicing).
    results = [RefereeResult.fail_open(label="main_judge", reason="invalid")]
    action = decide(
        results, [], primary_referee_label="main_judge", restructure_enabled=True
    )
    assert action.kind == "continue"
