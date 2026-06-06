"""Unit tests for the effect routes (change-2, Shape B).

Each route must reproduce ONE legacy decider branch (run_loop.py 840-906)
byte-for-byte: the same ``sm.transition_to`` / ``session.append_event`` / helper
calls in the same order, returning the matching ``Directive``. These cover the
transfer / customer_decline / restructure branches the golden net does not reach
(only goal_achieved is exercised by goal_achieved_wrapup).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isales_common.enums import CallStatus

from isales_engine.pipeline.decider import DeciderAction
from isales_engine.routes.effects import (
    CustomerDeclineRoute,
    EffectContext,
    GoalAchievedRoute,
    RestructureRoute,
    TransferRoute,
)
from isales_engine.select_router import Directive


class _FakeSM:
    def __init__(self) -> None:
        self.transitions: list[tuple[Any, Any]] = []

    def transition_to(self, state, reason=None, force=False):  # noqa: ANN001
        self.transitions.append((state, reason))


class _FakeSession:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.consecutive_restructure_count = 0
        self.consecutive_interruption_count = 0
        self.interruption_signaled = True
        self.wrap_up_started_at_monotonic: Any = None
        self.wrap_up_started_at_wallclock: Any = None

    def append_event(self, event_type: str, **fields: Any) -> None:
        self.events.append((event_type, fields))


@dataclass
class _WrapUp:
    max_rounds: int = 2
    max_seconds: int = 30


@dataclass
class _Pipeline:
    default_replies: list[str] = field(default_factory=list)


@dataclass
class _Config:
    voice_id: str = "default"
    wrap_up: _WrapUp = field(default_factory=_WrapUp)
    pipeline: _Pipeline = field(default_factory=_Pipeline)


class _Providers:
    tts = "TTS"


def _ctx(
    session: _FakeSession,
    sm: _FakeSM,
    *,
    action: DeciderAction,
    restructure_text: str | None = None,
    restructure_capped: bool = False,
    goal_type: str = "",
    config: _Config | None = None,
) -> tuple[EffectContext, dict]:
    calls: dict[str, Any] = {}

    async def perform_handoff(*a: Any, **k: Any) -> None:
        calls["handoff"] = (a, k)

    async def run_restructure(*a: Any, **k: Any) -> bool:
        calls["restructure"] = (a, k)
        return calls.get("restructure_played", True)

    async def play_tts(*a: Any, **k: Any) -> None:
        calls.setdefault("tts", []).append((a, k))

    ctx = EffectContext(
        session=session,
        sm=sm,
        telephony="TEL",
        providers=_Providers(),
        config=config or _Config(),
        action=action,
        restructure_text=restructure_text,
        restructure_capped=restructure_capped,
        goal_type=goal_type,
        pipeline_timeout_ms=8000,
        perform_handoff=perform_handoff,
        run_restructure=run_restructure,
        play_tts=play_tts,
        now_utc=lambda: "WALL",
    )
    return ctx, calls


async def test_goal_achieved_route() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, _ = _ctx(
        session,
        sm,
        action=DeciderAction(kind="transition", to="goal_achieved", goal_type="appointment"),
        goal_type="appointment",
    )
    assert await GoalAchievedRoute().execute(ctx) is Directive.CONTINUE
    assert sm.transitions == [(CallStatus.WRAPPING_UP, "appointment")]
    assert [e[0] for e in session.events] == ["wrap_up_started", "goal_achieved"]
    assert session.wrap_up_started_at_wallclock == "WALL"


async def test_goal_achieved_route_empty_goal_type_uses_default_reason() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, _ = _ctx(
        session, sm, action=DeciderAction(kind="transition", to="goal_achieved"), goal_type=""
    )
    await GoalAchievedRoute().execute(ctx)
    assert sm.transitions == [(CallStatus.WRAPPING_UP, "goal_achieved")]


async def test_transfer_route() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, calls = _ctx(session, sm, action=DeciderAction(kind="transition", to="transfer"))
    assert await TransferRoute().execute(ctx) is Directive.RETURN
    assert "handoff" in calls
    _, kw = calls["handoff"]
    assert kw["trigger_type"] == "referee"
    assert kw["trigger_detail"] == "referee_decision"
    assert kw["phrase"] == ""
    assert sm.transitions == []  # handoff itself drives transitions, not the route


async def test_customer_decline_route() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, _ = _ctx(session, sm, action=DeciderAction(kind="transition", to="customer_decline"))
    assert await CustomerDeclineRoute().execute(ctx) is Directive.CONTINUE
    assert sm.transitions == [
        (CallStatus.ACTIVATING, "customer_decline_recovery"),
        (CallStatus.LISTENING, "customer_decline_done"),
    ]


async def test_restructure_capped_with_default_reply() -> None:
    sm, session = _FakeSM(), _FakeSession()
    session.consecutive_restructure_count = 5
    cfg = _Config(pipeline=_Pipeline(default_replies=["稍等"]))
    ctx, calls = _ctx(
        session, sm, action=DeciderAction(kind="restructure"), restructure_capped=True, config=cfg
    )
    assert await RestructureRoute().execute(ctx) is Directive.CONTINUE
    assert calls.get("tts")  # default reply played
    assert session.consecutive_restructure_count == 0
    assert sm.transitions == [(CallStatus.LISTENING, "restructure_capped")]


async def test_restructure_capped_no_default_reply() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, calls = _ctx(
        session, sm, action=DeciderAction(kind="restructure"), restructure_capped=True
    )
    assert await RestructureRoute().execute(ctx) is Directive.CONTINUE
    assert "tts" not in calls
    assert sm.transitions == [(CallStatus.LISTENING, "restructure_capped")]


async def test_restructure_text_played() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, calls = _ctx(
        session, sm, action=DeciderAction(kind="restructure"), restructure_text="再说一遍"
    )
    calls["restructure_played"] = True
    assert await RestructureRoute().execute(ctx) is Directive.CONTINUE
    assert session.consecutive_restructure_count == 1
    assert sm.transitions == [(CallStatus.LISTENING, "restructure_done")]


async def test_restructure_text_interrupted() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, calls = _ctx(
        session, sm, action=DeciderAction(kind="restructure"), restructure_text="再说一遍"
    )
    calls["restructure_played"] = False
    assert await RestructureRoute().execute(ctx) is Directive.CONTINUE
    assert session.consecutive_restructure_count == 1
    assert session.consecutive_interruption_count == 1
    assert session.interruption_signaled is False
    assert sm.transitions == [(CallStatus.INTERRUPTED, "restructure_interrupted")]


async def test_restructure_degraded_falls_through() -> None:
    sm, session = _FakeSM(), _FakeSession()
    ctx, _ = _ctx(
        session,
        sm,
        action=DeciderAction(kind="restructure"),
        restructure_text=None,
        restructure_capped=False,
    )
    assert await RestructureRoute().execute(ctx) is Directive.FALL_THROUGH
    assert sm.transitions == []
