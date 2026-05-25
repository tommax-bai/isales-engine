"""Unit tests for the ARTC driver thread.

Spec: device-hardware § Requirement: 云端 isales-engine
``transport/aliyun_rtc.py`` 通过专用 SDK 驱动线程承载 vendor recvCoroutine.

These tests exercise :class:`_ArtcDriverThread` in isolation — no vendor
SDK involvement. Vendor integration tests live in
``test_aliyun_rtc_thread_model.py`` (with a stub for ``AliRTCEngine``).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from isales_engine.transport._rtc_driver import (
    DriverQueueFull,
    _ArtcDriverThread,
    _DEFAULT_PUMP_INTERVAL_S,
    _read_pump_interval,
)


# --------------------------------------------------------------------------
# pump_interval env override
# --------------------------------------------------------------------------


def test_default_pump_interval_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISALES_ARTC_PUMP_INTERVAL_MS", raising=False)
    assert _read_pump_interval() == _DEFAULT_PUMP_INTERVAL_S


def test_pump_interval_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "50")
    assert _read_pump_interval() == pytest.approx(0.050)


def test_pump_interval_env_garbage_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "not-a-number")
    assert _read_pump_interval() == _DEFAULT_PUMP_INTERVAL_S


def test_pump_interval_env_zero_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ARTC_PUMP_INTERVAL_MS", "0")
    assert _read_pump_interval() == _DEFAULT_PUMP_INTERVAL_S


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_then_stop_thread_cleanly() -> None:
    drv = _ArtcDriverThread(name="t-lifecycle", pump_interval_s=0.01)
    assert drv.is_alive is False
    drv.start(main_loop=asyncio.get_running_loop())
    try:
        assert drv.is_alive is True
        # Let at least one pump iteration run so the loop is actually
        # being driven (would have failed if set_event_loop did not happen).
        await asyncio.sleep(0.05)
        assert drv.is_healthy is True
    finally:
        await asyncio.to_thread(drv.stop)
    assert drv.is_alive is False


@pytest.mark.asyncio
async def test_start_twice_raises() -> None:
    drv = _ArtcDriverThread(name="t-double-start", pump_interval_s=0.01)
    drv.start(main_loop=asyncio.get_running_loop())
    try:
        with pytest.raises(RuntimeError, match="already started"):
            drv.start(main_loop=asyncio.get_running_loop())
    finally:
        await asyncio.to_thread(drv.stop)


@pytest.mark.asyncio
async def test_stop_idempotent() -> None:
    drv = _ArtcDriverThread(name="t-stop-idemp", pump_interval_s=0.01)
    drv.start(main_loop=asyncio.get_running_loop())
    await asyncio.to_thread(drv.stop)
    # Second stop is a no-op (thread reference cleared).
    await asyncio.to_thread(drv.stop)


# --------------------------------------------------------------------------
# call() / submit()
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_runs_on_driver_thread_and_returns_result() -> None:
    drv = _ArtcDriverThread(name="t-call", pump_interval_s=0.01)
    drv.start(main_loop=asyncio.get_running_loop())
    try:
        observed_thread: list[str] = []

        def work(x: int, y: int) -> int:
            observed_thread.append(threading.current_thread().name)
            return x + y

        result = await drv.call(work, 2, 3)
        assert result == 5
        # MUST have run on the driver thread, not the caller's thread.
        assert observed_thread == ["t-call"]
    finally:
        await asyncio.to_thread(drv.stop)


@pytest.mark.asyncio
async def test_call_propagates_exception_from_driver_thread() -> None:
    drv = _ArtcDriverThread(name="t-exc", pump_interval_s=0.01)
    drv.start(main_loop=asyncio.get_running_loop())
    try:
        def bad() -> None:
            raise ValueError("vendor said no")

        with pytest.raises(ValueError, match="vendor said no"):
            await drv.call(bad)
        # An exception in a submitted callable MUST NOT kill the driver
        # thread — subsequent submits keep working.
        assert drv.is_alive is True
        assert await drv.call(lambda: 7) == 7
    finally:
        await asyncio.to_thread(drv.stop)


@pytest.mark.asyncio
async def test_submit_before_start_raises() -> None:
    drv = _ArtcDriverThread(name="t-not-started", pump_interval_s=0.01)
    with pytest.raises(RuntimeError, match="not started"):
        drv.submit(lambda: 1)


@pytest.mark.asyncio
async def test_submit_after_stop_raises() -> None:
    drv = _ArtcDriverThread(name="t-stopped", pump_interval_s=0.01)
    drv.start(main_loop=asyncio.get_running_loop())
    await asyncio.to_thread(drv.stop)
    with pytest.raises(RuntimeError, match="not started"):
        drv.submit(lambda: 1)


# --------------------------------------------------------------------------
# Backpressure / queue cap
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_cap_raises_driver_queue_full() -> None:
    # Small cap + a blocked driver thread → next submit must fail loudly
    # rather than block. We block the driver thread by submitting a long
    # sleep first, then filling the queue while it's busy.
    drv = _ArtcDriverThread(
        name="t-cap",
        pump_interval_s=0.01,
        cmd_queue_cap=2,
    )
    drv.start(main_loop=asyncio.get_running_loop())
    try:
        block = threading.Event()

        def blocker() -> None:
            block.wait(timeout=2.0)

        # Occupy the driver thread.
        blocker_fut = drv.submit(blocker)
        # Wait until the driver thread has picked the blocker off the queue.
        # Brief poll — the drain runs at <= pump interval (10ms).
        deadline = time.monotonic() + 1.0
        while drv.queue_size() != 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert drv.queue_size() == 0

        # Fill the queue to its cap (2 items).
        f1 = drv.submit(lambda: 1)
        f2 = drv.submit(lambda: 2)
        assert not f1.done() and not f2.done()
        assert drv.queue_size() == 2

        # Next submit MUST raise rather than block.
        with pytest.raises(DriverQueueFull):
            drv.submit(lambda: 3)

        # Unblock the driver — pending futures should resolve.
        block.set()
        assert await asyncio.wait_for(f1, timeout=1.0) == 1
        assert await asyncio.wait_for(f2, timeout=1.0) == 2
        await asyncio.wait_for(blocker_fut, timeout=1.0)
    finally:
        block.set()
        await asyncio.to_thread(drv.stop)


# --------------------------------------------------------------------------
# Shutdown semantics
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_fails_pending_futures_with_clear_error() -> None:
    """If a driver thread is asked to stop while commands are queued,
    those futures resolve with a ``RuntimeError("...stopping")`` rather
    than dangling forever or returning a stale success.

    Timing matters: we need the stop flag to flip BEFORE the driver
    finishes its in-flight cmd, otherwise the pending cmds get drained
    normally as the driver flies through its inner loop. The test below
    sequences this explicitly: ``drv.stop()`` runs in a thread, sets the
    flag, then blocks on ``thread.join`` (because the driver is still
    stuck in the blocker). Once we observe the flag set, we release the
    blocker — the driver then sees stop set and fails the pending cmds
    via :meth:`_drain_commands_once`.
    """
    drv = _ArtcDriverThread(
        name="t-pending-on-stop",
        pump_interval_s=0.05,
        cmd_queue_cap=8,
    )
    drv.start(main_loop=asyncio.get_running_loop())

    block = threading.Event()
    in_blocker = threading.Event()

    def blocker() -> None:
        in_blocker.set()
        block.wait(timeout=2.0)

    # Block the driver, wait until it's actually inside blocker, then
    # queue 3 cmds while the driver is pinned.
    blocker_fut = drv.submit(blocker)
    deadline = time.monotonic() + 1.0
    while not in_blocker.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert in_blocker.is_set()
    pending_futs = [drv.submit(lambda i=i: i) for i in range(3)]

    # Schedule stop on a worker thread. It will set _stop.is_set()
    # immediately, then block on thread.join (because the driver is
    # still stuck in blocker).
    stop_task = asyncio.create_task(asyncio.to_thread(drv.stop))

    # Wait until the stop flag has actually flipped before unblocking —
    # otherwise the driver finishes blocker and drains the pending cmds
    # normally before stop has had a chance to set the flag.
    deadline = time.monotonic() + 1.0
    while not drv._stop.is_set() and time.monotonic() < deadline:  # noqa: SLF001
        await asyncio.sleep(0.005)
    assert drv._stop.is_set()  # noqa: SLF001

    # Now release the blocker. The driver finishes blocker normally
    # (it was in-flight), then sees stop set and fails the 3 pending.
    block.set()
    await stop_task

    # Blocker completed normally (it was already executing).
    await asyncio.wait_for(blocker_fut, timeout=1.0)

    # Pending ones got the shutdown exception.
    for fut in pending_futs:
        with pytest.raises(RuntimeError, match="stopping"):
            await asyncio.wait_for(fut, timeout=1.0)


@pytest.mark.asyncio
async def test_thread_is_daemon() -> None:
    drv = _ArtcDriverThread(name="t-daemon", pump_interval_s=0.01)
    drv.start(main_loop=asyncio.get_running_loop())
    try:
        assert drv._thread is not None  # noqa: SLF001 — verifying lifecycle
        assert drv._thread.daemon is True  # noqa: SLF001
    finally:
        await asyncio.to_thread(drv.stop)
