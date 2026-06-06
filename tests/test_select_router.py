"""Unit tests for the SelectRouter core (change-2, Shape B).

Covers the selector mapping (DeciderAction → effect-route id), dispatch routing,
the None→FALL_THROUGH path, and that the Router does NOT swallow route
exceptions (the legacy inline path lets them propagate; byte identity requires
the same).
"""

from __future__ import annotations

import pytest

from isales_engine.pipeline.decider import DeciderAction
from isales_engine.routes.selector import select_effect_route
from isales_engine.select_router import Directive, Router


def test_selector_maps_transition_targets() -> None:
    assert (
        select_effect_route(DeciderAction(kind="transition", to="goal_achieved"))
        == "goal_achieved"
    )
    assert (
        select_effect_route(DeciderAction(kind="transition", to="transfer"))
        == "transfer"
    )
    assert (
        select_effect_route(DeciderAction(kind="transition", to="customer_decline"))
        == "customer_decline"
    )


def test_selector_maps_restructure() -> None:
    assert (
        select_effect_route(DeciderAction(kind="restructure", source="last_reply"))
        == "restructure"
    )


def test_selector_continue_and_unknown_fall_through() -> None:
    # continue-action and unknown transition targets have no effect route →
    # caller falls through to the shared LISTENING tail (legacy 908/937).
    assert select_effect_route(DeciderAction(kind="continue")) is None
    assert (
        select_effect_route(DeciderAction(kind="transition", to="something_else"))
        is None
    )


class _FakeRoute:
    def __init__(self, route_id: str, directive: Directive) -> None:
        self.route_id = route_id
        self.metadata: dict = {}
        self._directive = directive
        self.calls: list = []

    async def execute(self, ctx):  # noqa: ANN001
        self.calls.append(ctx)
        return self._directive


async def test_dispatch_runs_selected_route() -> None:
    route = _FakeRoute("transfer", Directive.RETURN)
    router = Router({"transfer": route}, lambda a: "transfer")
    out = await router.dispatch(
        DeciderAction(kind="transition", to="transfer"), "CTX"
    )
    assert out is Directive.RETURN
    assert route.calls == ["CTX"]


async def test_dispatch_none_selector_falls_through() -> None:
    router = Router({}, lambda a: None)
    out = await router.dispatch(DeciderAction(kind="continue"), "CTX")
    assert out is Directive.FALL_THROUGH


async def test_dispatch_does_not_swallow_route_exception() -> None:
    class _Boom:
        route_id = "boom"
        metadata: dict = {}

        async def execute(self, ctx):  # noqa: ANN001
            raise RuntimeError("boom")

    router = Router({"boom": _Boom()}, lambda a: "boom")
    with pytest.raises(RuntimeError, match="boom"):
        await router.dispatch(DeciderAction(kind="continue"), None)
