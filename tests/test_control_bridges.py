"""change-1 (engine-eventbus-foundation §2.6) — control-plane bridges.

The old manual-hangup mechanism (``request_manual_hangup`` → ``main.cancel()``
sentinel, with a never-raised ``_ManualHangupRequested``) had ZERO test
coverage — ``test_event_pubsub`` only checks that the EngineControl consumer
calls the injected handlers; the handler bodies (main.py) + the run_loop
mechanism were untested.

2.6 replaces that with a bus bridge: main.py's ``_on_manual_hangup`` /
``_on_transfer`` ``bus.post`` a ``ManualHangupRequested`` / ``TransferRequested``;
``_subscribe_control_bridges`` terminates the call by cancelling the run-loop
``main`` task with ``MANUAL_HANGUP`` — byte-identical to the deleted sentinel.
These tests pin that mechanism (the new behavioral path).
"""

from __future__ import annotations

import asyncio

from isales_common.enums import HangupCause

from isales_engine.events import ManualHangupRequested, TransferRequested
from isales_engine.run_loop import _subscribe_control_bridges
from tests.test_run_loop import _make_session


async def _bus_with_bridges():
    # The session is constructed WITH an (unstarted) bus; mirror run_session:
    # subscribe the control bridge onto session.bus, then start it.
    session = _make_session()
    bus = session.bus
    _subscribe_control_bridges(session, bus)
    await bus.start()
    return bus, session


async def _spawn_main(session) -> asyncio.Task:
    async def _main() -> None:
        await asyncio.sleep(60)

    task = asyncio.create_task(_main(), name="main")
    session.tasks["main"] = task
    return task


async def _wait_cause(session) -> None:
    for _ in range(100):
        if session.hangup_cause is not None:
            return
        await asyncio.sleep(0.01)


async def test_manual_hangup_event_cancels_main_with_manual_cause() -> None:
    """ManualHangupRequested → bridge sets MANUAL_HANGUP then cancels 'main'
    (byte-identical to the deleted request_manual_hangup)."""
    bus, session = await _bus_with_bridges()
    main_task = await _spawn_main(session)
    try:
        bus.post(ManualHangupRequested(operator="alice"))
        await _wait_cause(session)
        await asyncio.gather(main_task, return_exceptions=True)
        assert session.hangup_cause == HangupCause.MANUAL_HANGUP.value
        assert main_task.cancelled()
    finally:
        await bus.aclose()


async def test_transfer_event_routes_through_hangup_stage4_stub() -> None:
    """TransferRequested currently routes through the same hangup path (stage-4
    stub: _on_transfer always set MANUAL_HANGUP). Byte-identical preservation."""
    bus, session = await _bus_with_bridges()
    main_task = await _spawn_main(session)
    try:
        bus.post(TransferRequested(agent_id="42"))
        await _wait_cause(session)
        await asyncio.gather(main_task, return_exceptions=True)
        assert session.hangup_cause == HangupCause.MANUAL_HANGUP.value
        assert main_task.cancelled()
    finally:
        await bus.aclose()


async def test_control_bridge_noop_when_no_main_task() -> None:
    """No 'main' task registered → bridge is a no-op (matches the old
    `if main is None ... return` guard). No error, no cause set."""
    bus, session = await _bus_with_bridges()
    try:
        bus.post(ManualHangupRequested(operator=None))
        await asyncio.sleep(0.05)
        assert session.hangup_cause is None
    finally:
        await bus.aclose()


async def test_control_bridge_noop_when_main_already_done() -> None:
    """'main' already finished → bridge does not set a cause (matches the old
    `or main.done()` guard)."""
    bus, session = await _bus_with_bridges()

    async def _done() -> None:
        return

    main_task = asyncio.create_task(_done(), name="main")
    await main_task
    session.tasks["main"] = main_task
    try:
        bus.post(ManualHangupRequested(operator=None))
        await asyncio.sleep(0.05)
        assert session.hangup_cause is None
    finally:
        await bus.aclose()
