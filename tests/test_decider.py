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
# engine-auto-restructure-on-interrupt: the restructure routing action was
# removed; a route-to-recovery rule stands in for the ordering tests.
RECOVERY_RULE = {
    "referee": "intent",
    "match": ["NEGATIVE"],
    "action": {"type": "route", "to": "recovery"},
}


def test_no_referees_continue():
    action = decide([], [GOAL_RULE])
    assert action.kind == "continue"
    assert action.matched_rule is None


def test_first_match_wins():
    # Both reject→transfer and intent→recovery would match; transfer is first.
    results = [_r("reject", "OPERATOR"), _r("intent", "NEGATIVE")]
    rules = [TRANSFER_RULE, RECOVERY_RULE]
    action = decide(results, rules)
    assert action.kind == "transition"
    assert action.to == "transfer"
    assert action.matched_rule == TRANSFER_RULE


def test_order_matters_recovery_first():
    results = [_r("reject", "OPERATOR"), _r("intent", "NEGATIVE")]
    rules = [RECOVERY_RULE, TRANSFER_RULE]
    action = decide(results, rules)
    assert action.kind == "route"
    assert action.to == "recovery"


def test_goal_achieved_carries_goal_type():
    action = decide([_r("main_judge", "goal_achieved")], [GOAL_RULE])
    assert action.kind == "transition"
    assert action.to == "goal_achieved"
    assert action.goal_type == "appointment"


def test_route_closing_carries_goal_type():
    # fix-goal-achievement-pipeline: the modern `route to=closing` action carries
    # goal_type symmetrically to the legacy `transition` action above. Regression
    # for the decider dropping goal_type on the route branch (→ empty goal_type).
    rule = {
        "referee": "main_judge",
        "match": ["SUCCESS"],
        "action": {
            "type": "route",
            "to": "closing",
            "then_state": "WRAPPING_UP",
            "goal_type": "intent_confirmed",
        },
    }
    action = decide([_r("main_judge", "SUCCESS")], [rule])
    assert action.kind == "route"
    assert action.to == "closing"
    assert action.then_state == "WRAPPING_UP"
    assert action.goal_type == "intent_confirmed"


def test_route_without_goal_type_is_none():
    rule = {
        "referee": "j",
        "match": ["GO"],
        "action": {"type": "route", "to": "persona:foo"},
    }
    action = decide([_r("j", "GO")], [rule])
    assert action.kind == "route"
    assert action.goal_type is None


def test_no_match_continue():
    action = decide([_r("main_judge", "continue")], [GOAL_RULE])
    assert action.kind == "continue"


def test_confidence_ignored_category_drives_match():
    # engine-tools-multidialogue-gating: the confidence floor is gone (referees
    # pin confidence=1.0). The bare category alone drives matching — a stray
    # low-confidence value is ignored, not a no-match.
    action = decide([_r("main_judge", "goal_achieved", confidence=0.5)], [GOAL_RULE])
    assert action.kind == "transition"
    assert action.to == "goal_achieved"


def test_failopen_referee_does_not_match():
    action = decide([RefereeResult.fail_open(label="main_judge", reason="timeout")], [GOAL_RULE])
    assert action.kind == "continue"


def test_low_confidence_no_longer_triggers_restructure():
    # engine-interruption-rule-tree D6: the low-confidence→restructure fallback
    # was removed (dead code — referees hardcode confidence=1.0). A referee whose
    # category matches no rule falls through to fail-open continue, never
    # restructure, and never a "low_confidence" trigger. (The confidence value is
    # now ignored entirely; only the bare category drives matching.)
    results = [_r("main_judge", "continue", confidence=0.4)]
    action = decide(results, [GOAL_RULE])
    assert action.kind == "continue"
    assert action.restructure_trigger is None


def test_no_rule_match_falls_through_to_continue():
    # A hard fail-open (category None) on the only referee → continue.
    results = [RefereeResult.fail_open(label="main_judge", reason="invalid")]
    action = decide(results, [])
    assert action.kind == "continue"


def test_decider_stays_pure_no_auto_restructure_knowledge():
    # engine-auto-restructure-on-interrupt: decide() has NO session / switch /
    # interrupt_remaining_text awareness — it always returns continue on no-match.
    # The auto-restructure override lives ONLY at the run_loop call site, so
    # decide()'s contract is unchanged regardless of campaign switches.
    results = [_r("main_judge", "continue", confidence=1.0)]
    action = decide(results, [])
    assert action.kind == "continue"
    assert action.matched_rule is None


def test_tool_action_carries_per_rule_closing_phrase():
    # §11: a tool rule's per-keyword closing_phrase rides on the DeciderAction so
    # one hangup tool can be reused across keywords with different phrases.
    rule = {
        "referee": "j",
        "match": ["OFFENSIVE"],
        "action": {"type": "tool", "tool": "bye", "closing_phrase": "不打扰了，再见"},
    }
    action = decide([_r("j", "OFFENSIVE")], [rule])
    assert action.kind == "tool"
    assert action.tool == "bye"
    assert action.closing_phrase == "不打扰了，再见"


def test_tool_action_without_closing_phrase_is_none():
    rule = {"referee": "j", "match": ["HANGUP"], "action": {"type": "tool", "tool": "bye"}}
    action = decide([_r("j", "HANGUP")], [rule])
    assert action.kind == "tool"
    assert action.closing_phrase is None
