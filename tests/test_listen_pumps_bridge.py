"""change-1 (engine-eventbus-foundation §2) — listen-pump → EventBus → bridge
wiring.

The golden-transcript net and the wider run_session suite prove the CONSUMER
(``_main_turn_loop``) is byte-identical, but they exercise only LISTENING-shaped
flows: the golden fixtures contain no ``interruption`` event, ``test_eventbus``
uses ``drain_inline`` (a single coroutine, never the two concurrent
dispatchers), and ``test_realtime_interruption`` injects into a hand-built
``asr_partials_q`` that bypasses ``bus.post`` / the bridge entirely.

So nothing else proves that a signal POSTED by the real ``_asr_pump`` /
``_ev_pump`` actually reaches the existing queues/events THROUGH the real
two-lane dispatchers + bridge subscribers. These tests pin exactly that — the
new code path — deterministically (awaited ``.get()`` / ``.wait()``, no barge-in
wall-clock race).

The per-call EventBus is created + started + owned by ``run_session`` (so the
control-plane hangup bridge is live from dial); ``_start_listen_pumps`` only
wires the ASR producers/bridges onto ``session.bus``. These tests mirror that
setup: create + start the bus, run the pumps, then ``_stop_listen_pumps`` +
``bus.aclose()`` (the teardown order run_session uses).
"""

from __future__ import annotations

import asyncio

from isales_engine.call_session import CallSession
from isales_engine.eventbus import Lane
from isales_engine.events import AsrFinal, AsrPartial
from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import _start_listen_pumps, _stop_listen_pumps
from tests.test_run_loop import _make_config, _make_providers, _make_session


async def _setup() -> tuple[MockTelephonyClient, CallSession]:
    """Mirror run_session's bus setup: the session is constructed WITH an
    (unstarted) bus (CallSession.bus factory); run_session start()s it. Here we
    just start it, then dial-connect the mock telephony."""
    session = _make_session()
    await session.bus.start()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await tel.dial(session.call_record_id, "+x")  # emits "connected"
    return tel, session


async def _teardown(session: CallSession, ctx: object) -> None:
    """Mirror run_session teardown: stop the listen pumps, then aclose the bus
    (the bus outlives the pumps — run_session owns its aclose)."""
    await _stop_listen_pumps(session, ctx)  # type: ignore[arg-type]
    assert session.bus is not None
    await session.bus.aclose()


async def test_lanes_registered_partial_lossy_final_lossless() -> None:
    """The session wiring must register AsrPartial LOSSY; finals / hangup /
    stream-end stay LOSSLESS (a user turn must never be silently dropped)."""
    tel, session = await _setup()
    providers = _make_providers(ScriptedMockASR(partial_step_ms=10))
    ctx = await _start_listen_pumps(session, _make_config(), tel, providers)
    try:
        assert session.bus is not None
        assert session.bus.lane_for(AsrPartial) is Lane.LOSSY
        assert session.bus.lane_for(AsrFinal) is Lane.LOSSLESS
    finally:
        await _teardown(session, ctx)


async def test_asr_pump_posts_reach_queues_through_real_dispatchers() -> None:
    """A fed ASR turn flows _asr_pump → bus.post → (real lossy + lossless
    dispatchers) → bridge → asr_partials_q / asr_finals_q. The bridge
    reconstructs ASRResult with the correct is_final, and the consumers (which
    read only .text) get byte-identical text."""
    tel, session = await _setup()
    asr = ScriptedMockASR(partial_step_ms=10)
    ctx = await _start_listen_pumps(session, _make_config(), tel, _make_providers(asr))
    try:
        await asr.feed_turn("你好吗")

        # Lossy lane delivers the partials (first partial of "你好吗" is "你").
        partial = await asyncio.wait_for(ctx.asr_partials_q.get(), timeout=1.0)
        assert partial.is_final is False
        assert partial.text and "你好吗".startswith(partial.text)

        # Lossless lane delivers the final, byte-identical text, is_final=True.
        final = await asyncio.wait_for(ctx.asr_finals_q.get(), timeout=1.0)
        assert final.is_final is True
        assert final.text == "你好吗"

        # The final also clears the partial-monitor's speech-start anchor in
        # the producer (preserved behavior).
        assert session.current_user_speech_started_ms is None

        # No lossy drop at human cadence — the audit counter stays clean.
        assert session.bus is not None and session.bus.dropped == 0
    finally:
        await _teardown(session, ctx)


async def test_ev_pump_hangup_sets_hangup_event_through_bridge() -> None:
    """A telephony remote_hangup flows _ev_pump → bus.post(HangupDetected) →
    bridge → hangup_event.set() (the only thing _await_user_or_silence checks)."""
    tel, session = await _setup()
    ctx = await _start_listen_pumps(
        session, _make_config(), tel, _make_providers(ScriptedMockASR(partial_step_ms=10))
    )
    try:
        assert not ctx.hangup_event.is_set()
        await tel.simulate_remote_hangup(session.call_record_id)
        await asyncio.wait_for(ctx.hangup_event.wait(), timeout=1.0)
        assert ctx.hangup_event.is_set()
    finally:
        await _teardown(session, ctx)


async def test_stop_pumps_drains_stream_ended_and_aclose_closes_bus() -> None:
    """Teardown: with the pump actually running (as in any real call),
    cancelling _asr_pump runs its finally → post(AsrStreamEnded) onto the
    still-open bus; run_session's bus.aclose() drains the lossless lane →
    bridge → asr_finished.set(); the bus ends closed and the session task
    registry is cleaned up by _stop_listen_pumps.

    (We feed a turn + drain the final first so the pump is past its initial
    await before teardown. Cancelling a not-yet-started task raises before its
    try/finally — so AsrStreamEnded would not post — but that edge is benign:
    asr_finished has no consumer, and a real call always runs the pump for the
    whole session before teardown.)"""
    tel, session = await _setup()
    asr = ScriptedMockASR(partial_step_ms=10)
    ctx = await _start_listen_pumps(session, _make_config(), tel, _make_providers(asr))
    assert "asr_pump" in session.tasks
    await asr.feed_turn("在的")
    final = await asyncio.wait_for(ctx.asr_finals_q.get(), timeout=1.0)
    assert final.text == "在的"  # pump is now genuinely running

    await _stop_listen_pumps(session, ctx)
    assert session.bus is not None
    await session.bus.aclose()

    assert ctx.asr_finished.is_set()  # AsrStreamEnded survived the cancel→drain
    assert session.bus._closed is True
    for key in ("asr_pump", "ev_pump", "partial_monitor", "vad_monitor"):
        assert key not in session.tasks


async def test_lossless_final_not_starved_by_partial_flood() -> None:
    """The reviewer's cross-lane concern, end-to-end: a burst of partials on
    the lossy lane (separate dispatcher) must never delay/lose a final on the
    lossless lane. Feed a long utterance (many partials) + final; the final
    must still arrive intact through the real dispatchers."""
    tel, session = await _setup()
    asr = ScriptedMockASR(partial_step_ms=1)
    ctx = await _start_listen_pumps(session, _make_config(), tel, _make_providers(asr))
    try:
        long_utterance = "我想详细了解一下你们这个产品的具体情况和价格"
        await asr.feed_turn(long_utterance)
        final = await asyncio.wait_for(ctx.asr_finals_q.get(), timeout=1.0)
        assert final.text == long_utterance
        assert final.is_final is True
    finally:
        await _teardown(session, ctx)
