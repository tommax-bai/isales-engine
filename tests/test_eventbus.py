"""Unit tests for the in-process two-lane :class:`EventBus` (change-1).

Lock the load-bearing properties: lane assignment, lossless-never-drops,
lossy-drop-oldest, per-handler exception isolation (one raising handler does NOT
kill the lane or sibling handlers), CancelledError propagation, and the
deterministic inline-drain test mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from isales_engine.eventbus import EventBus, Lane
from isales_engine.events import AsrFinal, AsrPartial, HangupDetected, VadFrame


def test_lane_defaults() -> None:
    bus = EventBus()
    # *Final / *Detected default LOSSLESS by name.
    assert bus.lane_for(AsrFinal) is Lane.LOSSLESS
    assert bus.lane_for(HangupDetected) is Lane.LOSSLESS

    @dataclass
    class WeirdThing:  # unknown suffix → safe default LOSSLESS (never silently drop)
        pass

    assert bus.lane_for(WeirdThing) is Lane.LOSSLESS

    # Explicit registration overrides.
    bus.register_lane(AsrPartial, Lane.LOSSY)
    bus.register_lane(VadFrame, Lane.LOSSY)
    assert bus.lane_for(AsrPartial) is Lane.LOSSY


async def test_subscribe_and_deliver_inline() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(AsrFinal, lambda e: seen.append(e.text))
    bus.post(AsrFinal(text="一"))
    bus.post(AsrFinal(text="二"))
    await bus.drain_inline()
    assert seen == ["一", "二"]


async def test_unsubscribe() -> None:
    bus = EventBus()
    seen: list[str] = []
    unsub = bus.subscribe(AsrFinal, lambda e: seen.append(e.text))
    bus.post(AsrFinal(text="一"))
    await bus.drain_inline()
    unsub()
    bus.post(AsrFinal(text="二"))
    await bus.drain_inline()
    assert seen == ["一"]


async def test_lossless_never_drops_under_flood() -> None:
    """The lossless lane is unbounded — 10k finals all survive."""
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(AsrFinal, lambda e: seen.append(e.text))
    for i in range(10_000):
        bus.post(AsrFinal(text=str(i)))
    await bus.drain_inline()
    assert len(seen) == 10_000
    assert bus.dropped == 0


async def test_lossy_drops_oldest_on_full() -> None:
    """The lossy lane is bounded drop-oldest — newest survive, count tracked."""
    bus = EventBus(lossy_maxsize=4)
    bus.register_lane(AsrPartial, Lane.LOSSY)
    seen: list[str] = []
    bus.subscribe(AsrPartial, lambda e: seen.append(e.text))
    for i in range(10):
        bus.post(AsrPartial(text=str(i)))
    await bus.drain_inline()
    # Only the last 4 survive; the other 6 were dropped-oldest.
    assert seen == ["6", "7", "8", "9"]
    assert bus.dropped == 6


async def test_handler_exception_isolated() -> None:
    """One raising handler must NOT kill the lane or sibling handlers."""
    bus = EventBus()
    seen: list[str] = []

    def bad(_e: AsrFinal) -> None:
        raise RuntimeError("boom (simulates the bridge.py RtcError pump-death bug)")

    bus.subscribe(AsrFinal, bad)
    bus.subscribe(AsrFinal, lambda e: seen.append(e.text))
    bus.post(AsrFinal(text="survives"))
    await bus.drain_inline()
    # Sibling handler still ran despite `bad` raising.
    assert seen == ["survives"]


async def test_async_handler_awaited() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def on_final(e: AsrFinal) -> None:
        await asyncio.sleep(0)
        seen.append(e.text)

    bus.subscribe(AsrFinal, on_final)
    bus.post(AsrFinal(text="async"))
    await bus.drain_inline()
    assert seen == ["async"]


async def test_cancelled_error_propagates() -> None:
    """A handler raising CancelledError must NOT be swallowed (barge-in self-cancel)."""
    bus = EventBus()

    def cancelling(_e: AsrFinal) -> None:
        raise asyncio.CancelledError

    bus.subscribe(AsrFinal, cancelling)
    bus.post(AsrFinal(text="x"))
    with pytest.raises(asyncio.CancelledError):
        await bus.drain_inline()


async def test_background_dispatch_then_aclose_drains_lossless() -> None:
    """With the real dispatcher running, posting delivers; aclose drains pending."""
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(AsrFinal, lambda e: seen.append(e.text))
    await bus.start()
    bus.post(AsrFinal(text="live"))
    # let the dispatcher run
    for _ in range(10):
        await asyncio.sleep(0)
        if seen:
            break
    assert seen == ["live"]
    # enqueue one more, then aclose must drain it before stopping
    bus.post(AsrFinal(text="drained"))
    await bus.aclose()
    assert seen == ["live", "drained"]
    # post after close is a no-op
    bus.post(AsrFinal(text="ignored"))
    assert seen == ["live", "drained"]
