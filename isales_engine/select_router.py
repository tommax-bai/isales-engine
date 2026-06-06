"""SelectRouter — effect-dispatch table for the flat refactor (change-2, Shape B).

Behind ``ENGINE_USE_ROUTER`` (default OFF; the golden-transcript net guards byte
identity). This routes ONLY the post-``decide()`` effect dispatch — the legacy
``_main_turn_loop`` if/elif at run_loop.py 840-906 (goal_achieved / transfer /
customer_decline / restructure). Dialogue generation, playback, the POST-reply
referees, ``decide()``, wrap-up handling, the restructure-cap-reset, and the
LISTENING tail all stay inline in run_loop and verbatim.

NOT in this change (all change-3): eager multi-persona dialogue routes (N>1) +
the live-generator deviation, lazy hangup/transfer tool routes, referee
pre-reply gating, and the StatusProjector. This module is the table change-3
extends with tool routes.

Removal trigger: change-3 (engine-tools-multidialogue-gating) Phase-4 deletes
the legacy inline effect dispatch + ``ENGINE_USE_ROUTER`` in one commit.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Directive(Enum):
    """What the turn loop does after an effect route runs — mirrors the legacy
    control flow exactly:

    * ``CONTINUE``     == the legacy ``continue`` (next loop iteration)
    * ``RETURN``       == the legacy ``return`` (end the call)
    * ``FALL_THROUGH`` == drop past the dispatch to the shared
      restructure-cap-reset + LISTENING tail (legacy continue-action / degraded
      restructure).
    """

    CONTINUE = "continue"
    RETURN = "return"
    FALL_THROUGH = "fall_through"


@runtime_checkable
class ExecutableRoute(Protocol):
    """An effect route: applies one decider-driven effect and returns a Directive.

    ``execute`` reproduces one legacy branch byte-for-byte (same transitions,
    events, and helper calls in the same order) via the helpers carried on
    ``ctx`` (see ``routes.effects.EffectContext``).
    """

    route_id: str
    metadata: dict[str, Any]

    async def execute(self, ctx: Any) -> Directive: ...


class Router:
    """Maps a ``DeciderAction`` to a registered effect route and runs it.

    The Router does NOT catch route exceptions — the legacy inline path lets
    them propagate, so byte identity requires the same. A selector returning
    ``None`` (continue-action / no match) yields ``Directive.FALL_THROUGH``.
    """

    def __init__(
        self,
        routes: dict[str, ExecutableRoute],
        selector: Callable[[Any], str | None],
    ) -> None:
        self._routes = routes
        self._selector = selector

    def select(self, action: Any) -> ExecutableRoute | None:
        route_id = self._selector(action)
        return self._routes[route_id] if route_id is not None else None

    async def dispatch(self, action: Any, ctx: Any) -> Directive:
        route = self.select(action)
        if route is None:
            return Directive.FALL_THROUGH
        return await route.execute(ctx)
