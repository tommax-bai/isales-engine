"""Selector — map a ``DeciderAction`` to an effect-route id (change-2, Shape B).

Mirrors the legacy if/elif dispatch at run_loop.py 840-906. ``None`` means
"no effect route" (continue-action / unknown) → the caller falls through to the
shared restructure-cap-reset + LISTENING tail, exactly as the legacy path does
when no branch matched.
"""

from __future__ import annotations

from isales_engine.pipeline.decider import DeciderAction


def select_effect_route(action: DeciderAction) -> str | None:
    if action.kind == "transition":
        if action.to == "goal_achieved":
            return "goal_achieved"
        if action.to == "transfer":
            return "transfer"
        if action.to == "customer_decline":
            return "customer_decline"
        return None  # unknown transition target → fall through (matches legacy)
    if action.kind == "restructure":
        return "restructure"
    return None  # continue / unknown → fall through
