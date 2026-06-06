"""Build the effect-route :class:`Router` (change-2, Shape B).

Registers the four effect routes + the selector. The Router is dormant unless
``ENGINE_USE_ROUTER`` is on (run_loop only builds it then). change-3 extends the
table with lazy hangup/transfer tool routes.
"""

from __future__ import annotations

from isales_engine.routes.effects import (
    CustomerDeclineRoute,
    GoalAchievedRoute,
    RestructureRoute,
    TransferRoute,
)
from isales_engine.routes.selector import select_effect_route
from isales_engine.select_router import ExecutableRoute, Router


def build_effect_router() -> Router:
    routes: dict[str, ExecutableRoute] = {
        "goal_achieved": GoalAchievedRoute(),
        "transfer": TransferRoute(),
        "customer_decline": CustomerDeclineRoute(),
        "restructure": RestructureRoute(),
    }
    return Router(routes, select_effect_route)
