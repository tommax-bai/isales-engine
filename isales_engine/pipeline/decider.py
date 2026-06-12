"""Routing-rule decider — turns N referee categories into one turn action.

Spec: ai-pipeline § "路由规则引擎（decider）" + § "重组流三触发场景".
engine-multi-referee-and-restructure D3 / D5.

The decider is pure: it takes the referees' results + the campaign's ordered
``routing_rules`` and returns a single :class:`DeciderAction`. It walks the
rules in order and the **first** rule whose bound referee returned a matching
category wins (first-match-wins). A referee with no usable category (fail-open)
never matches. No rule matching falls through to ``continue`` (fail-open, back
to LISTENING).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isales_engine.streaming.types import RefereeResult


@dataclass
class DeciderAction:
    """What run_loop should do this turn."""

    kind: str  # "transition" | "restructure" | "continue" | "route" | "tool"
    to: str | None = None  # transition target / route target (persona|closing|recovery|restructure)
    goal_type: str | None = None  # carried by goal_achieved transitions
    source: str | None = None  # restructure InterruptText source
    # "last_reply" / "interrupt_remaining" — for trace.
    restructure_trigger: str | None = None
    # engine-tools-multidialogue-gating: tool alias (RouteToolAction) + the
    # route's declared then_state side-effect (RoutePersonaAction/RouteToolAction)
    # the gate-first turn reads to project state (via the sole state writer).
    # Legacy transition/restructure leave these None.
    tool: str | None = None
    then_state: str | None = None
    # Per-rule hangup closing phrase (RouteToolAction.closing_phrase): a per-
    # keyword override of HangupToolConfig.closing_phrase so one hangup tool is
    # reused across keywords with different phrases. None → fall back to the tool
    # config (engine-tools-multidialogue-gating §11).
    closing_phrase: str | None = None
    # The matched routing rule (dict), for pipeline_trace; None when none matched.
    matched_rule: dict[str, Any] | None = None


def decide(
    referee_results: Sequence[RefereeResult],
    routing_rules: Sequence[dict[str, Any]],
) -> DeciderAction:
    """Run first-match-wins over ``routing_rules`` → a single action.

    ``decide()`` never returns ``kind="restructure"`` — restructure is no longer
    a routing action (engine-auto-restructure-on-interrupt). It is produced only
    by the ``campaign.auto_restructure_on_interrupt`` override at the run_loop
    decide() call site, not here.
    """
    categories = {r.label: r.effective_category() for r in referee_results}

    for rule in routing_rules:
        ref = rule.get("referee")
        cat = categories.get(ref)
        if cat is None:
            continue
        if cat not in (rule.get("match") or []):
            continue
        action = rule.get("action") or {}
        atype = action.get("type")
        if atype == "transition":
            return DeciderAction(
                kind="transition",
                to=action.get("to"),
                goal_type=action.get("goal_type"),
                matched_rule=rule,
            )
        # engine-tools-multidialogue-gating: new action types. The action→route
        # mapping widens (this is the only widening; decide()'s walk is unchanged).
        if atype == "route":
            return DeciderAction(
                kind="route",
                to=action.get("to"),
                # Symmetric with the legacy ``transition`` branch above: a
                # ``route to=closing`` action carries goal_type the same way
                # ``transition to=goal_achieved`` does, so the goal_achieved
                # event records the configured label instead of "" (goal-
                # achievement spec § "route 与 transition 动作的 goal_type 提取对称").
                goal_type=action.get("goal_type"),
                then_state=action.get("then_state"),
                matched_rule=rule,
            )
        if atype == "tool":
            return DeciderAction(
                kind="tool",
                tool=action.get("tool"),
                then_state=action.get("then_state"),
                closing_phrase=action.get("closing_phrase"),
                matched_rule=rule,
            )
        # Unknown action type → continue (defensive; api validates on write).
        return DeciderAction(kind="continue", matched_rule=rule)

    # No rule matched → fail-open continue (back to LISTENING). The former
    # low-confidence→restructure fallback was removed (engine-interruption-rule-
    # tree D6): it was dead code (referees hardcode confidence=1.0).
    return DeciderAction(kind="continue")
