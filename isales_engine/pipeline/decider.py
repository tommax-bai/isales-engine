"""Routing-rule decider — turns N referee categories into one turn action.

Spec: ai-pipeline § "路由规则引擎（decider）" + § "重组流三触发场景".
engine-multi-referee-and-restructure D3 / D5.

The decider is pure: it takes the referees' results + the campaign's ordered
``routing_rules`` and returns a single :class:`DeciderAction`. It walks the
rules in order and the **first** rule whose bound referee returned a matching
category wins (first-match-wins). A referee below the confidence floor, or with
no usable category, never matches. No rule matching falls through to a
low-confidence restructure (if the primary referee is uncertain and a
restructure stream is configured) or, finally, ``continue``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isales_engine.streaming.types import CONFIDENCE_THRESHOLD, RefereeResult


@dataclass
class DeciderAction:
    """What run_loop should do this turn."""

    kind: str  # "transition" | "restructure" | "continue" | "route" | "tool"
    to: str | None = None  # transition target / route target (persona|closing|recovery|restructure)
    goal_type: str | None = None  # carried by goal_achieved transitions
    source: str | None = None  # restructure InterruptText source
    # "last_reply" / "interrupt_remaining" / "low_confidence" — for trace.
    restructure_trigger: str | None = None
    # engine-tools-multidialogue-gating: tool alias (RouteToolAction) + the
    # route's declared then_state side-effect (RoutePersonaAction/RouteToolAction)
    # the StatusProjector reads. Legacy transition/restructure leave these None.
    tool: str | None = None
    then_state: str | None = None
    # The matched routing rule (dict), for pipeline_trace; None when none matched.
    matched_rule: dict[str, Any] | None = None


def decide(
    referee_results: Sequence[RefereeResult],
    routing_rules: Sequence[dict[str, Any]],
    *,
    primary_referee_label: str | None = None,
    restructure_enabled: bool = False,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> DeciderAction:
    """Run first-match-wins over ``routing_rules`` → a single action."""
    categories = {r.label: r.effective_category(threshold) for r in referee_results}

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
        if atype == "restructure":
            if not restructure_enabled:
                # D4: restructure action degrades to continue when no
                # restructure slot is configured.
                return DeciderAction(kind="continue", matched_rule=rule)
            source = action.get("source")
            return DeciderAction(
                kind="restructure",
                source=source,
                restructure_trigger=source,
                matched_rule=rule,
            )
        # engine-tools-multidialogue-gating: new action types. The action→route
        # mapping widens (this is the only widening; decide()'s walk is unchanged).
        if atype == "route":
            return DeciderAction(
                kind="route",
                to=action.get("to"),
                then_state=action.get("then_state"),
                matched_rule=rule,
            )
        if atype == "tool":
            return DeciderAction(
                kind="tool",
                tool=action.get("tool"),
                then_state=action.get("then_state"),
                matched_rule=rule,
            )
        # Unknown action type → continue (defensive; api validates on write).
        return DeciderAction(kind="continue", matched_rule=rule)

    # No rule matched. D5 (c): the primary referee returned a real (parseable)
    # category but is below the confidence floor → re-voice the last reply to
    # stall a turn rather than silently continuing.
    if restructure_enabled and primary_referee_label:
        primary = next(
            (r for r in referee_results if r.label == primary_referee_label), None
        )
        if (
            primary is not None
            and primary.category is not None
            and primary.confidence < threshold
        ):
            return DeciderAction(
                kind="restructure",
                source="last_reply",
                restructure_trigger="low_confidence",
            )

    return DeciderAction(kind="continue")
