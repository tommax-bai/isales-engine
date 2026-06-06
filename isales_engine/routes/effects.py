"""Effect routes (change-2, Shape B) — byte-identical wrappers over the existing
run_loop effect functions, dispatched by the SelectRouter when
``ENGINE_USE_ROUTER`` is on.

Each ``execute`` reproduces ONE legacy decider-driven branch (run_loop.py
840-906) exactly: the same ``sm.transition_to`` / ``session.append_event`` /
helper calls in the same order with the same arguments, returning the
``Directive`` that mirrors the branch's control flow (``continue`` / ``return``
/ fall-through). The run_loop helper functions (``_perform_handoff`` /
``_run_restructure`` / ``_play_tts``) and ``now_utc`` are injected via
``EffectContext`` so this module never imports run_loop (no import cycle).
"""

from __future__ import annotations

import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from isales_common.enums import CallStatus

from isales_engine.pipeline.decider import DeciderAction
from isales_engine.select_router import Directive


@dataclass
class EffectContext:
    """Per-turn payload + injected run_loop helpers an effect route needs.

    run_loop builds this at the dispatch site; the values mirror the locals the
    legacy inline branches read (run_loop.py 840-906).
    """

    session: Any
    sm: Any
    telephony: Any
    providers: Any
    config: Any
    action: DeciderAction
    restructure_text: str | None
    restructure_capped: bool
    goal_type: str
    pipeline_timeout_ms: int
    # Injected run_loop helpers (avoid a run_loop import cycle).
    perform_handoff: Callable[..., Awaitable[None]]
    run_restructure: Callable[..., Awaitable[bool]]
    play_tts: Callable[..., Awaitable[None]]
    now_utc: Callable[[], Any]


@dataclass
class GoalAchievedRoute:
    """transition→goal_achieved: WRAPPING_UP + wrap-up/goal events (=840-854)."""

    route_id: str = "goal_achieved"
    metadata: dict[str, Any] = field(default_factory=lambda: {"kind": "effect"})

    async def execute(self, ctx: EffectContext) -> Directive:
        ctx.sm.transition_to(CallStatus.WRAPPING_UP, reason=ctx.goal_type or "goal_achieved")
        ctx.session.wrap_up_started_at_monotonic = time.monotonic()
        ctx.session.wrap_up_started_at_wallclock = ctx.now_utc()
        ctx.session.append_event(
            "wrap_up_started",
            rounds_remaining=ctx.config.wrap_up.max_rounds,
            seconds_remaining=ctx.config.wrap_up.max_seconds,
        )
        ctx.session.append_event("goal_achieved", goal_type=ctx.goal_type, extracted={})
        return Directive.CONTINUE


@dataclass
class TransferRoute:
    """transition→transfer: decider-driven handoff, no extra phrase (=858-865)."""

    route_id: str = "transfer"
    metadata: dict[str, Any] = field(default_factory=lambda: {"kind": "effect"})

    async def execute(self, ctx: EffectContext) -> Directive:
        await ctx.perform_handoff(
            ctx.session,
            ctx.sm,
            ctx.telephony,
            ctx.providers.tts,
            trigger_type="referee",
            trigger_detail="referee_decision",
            phrase="",
            voice_id=ctx.config.voice_id,
        )
        return Directive.RETURN


@dataclass
class CustomerDeclineRoute:
    """transition→customer_decline: ACTIVATING→LISTENING recovery (=867-872)."""

    route_id: str = "customer_decline"
    metadata: dict[str, Any] = field(default_factory=lambda: {"kind": "effect"})

    async def execute(self, ctx: EffectContext) -> Directive:
        ctx.sm.transition_to(CallStatus.ACTIVATING, reason="customer_decline_recovery")
        ctx.sm.transition_to(CallStatus.LISTENING, reason="customer_decline_done")
        return Directive.CONTINUE


@dataclass
class RestructureRoute:
    """restructure: capped→default+LISTENING / text→run+INTERRUPTED|LISTENING /
    degraded→fall-through (=874-906)."""

    route_id: str = "restructure"
    metadata: dict[str, Any] = field(default_factory=lambda: {"kind": "effect"})

    async def execute(self, ctx: EffectContext) -> Directive:
        if ctx.restructure_capped:
            # D6: consecutive cap reached — stop re-voicing; play a default
            # reply (if any) and reset the counter.
            if ctx.config.pipeline.default_replies:
                await ctx.play_tts(
                    ctx.session,
                    ctx.telephony,
                    ctx.providers.tts,
                    random.choice(ctx.config.pipeline.default_replies),
                    voice_id=ctx.config.voice_id,
                )
            ctx.session.consecutive_restructure_count = 0
            ctx.sm.transition_to(CallStatus.LISTENING, reason="restructure_capped")
            return Directive.CONTINUE
        if ctx.restructure_text is not None:
            played_rs = await ctx.run_restructure(
                ctx.session,
                ctx.telephony,
                ctx.providers,
                ctx.config,
                ctx.restructure_text,
                pipeline_timeout_ms=ctx.pipeline_timeout_ms,
            )
            ctx.session.consecutive_restructure_count += 1
            if not played_rs:
                ctx.session.consecutive_interruption_count += 1
                ctx.session.interruption_signaled = False
                ctx.sm.transition_to(CallStatus.INTERRUPTED, reason="restructure_interrupted")
                return Directive.CONTINUE
            ctx.sm.transition_to(CallStatus.LISTENING, reason="restructure_done")
            return Directive.CONTINUE
        # restructure degraded (no InterruptText available) → continue (fall through
        # to the shared restructure-cap-reset + LISTENING tail).
        return Directive.FALL_THROUGH
