"""engine-greeting-window-prewarm — warm turn-1 downstream connections during
the greeting playout window.

These pin the three behavioral contracts of the change:

* ASR pumps start BEFORE the greeting (``start_monitors=False``) so the vendor
  connection is warmed during the greeting, while the two barge-in monitors are
  gated until :func:`_start_monitors` runs AFTER the greeting (greeting stays
  non-interruptible).
* Greeting-era ASR results are drained (:func:`_drain_greeting_era_asr`) before
  the monitors + main loop start, so greeting-era audio is never consumed.
* The main-LLM connection is pre-warmed only for the FIXED-template greeting (an
  LLM greeting warms the same main-slot client itself); the probe fail-softs.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider

from isales_engine.call_session import CallSession
from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import (
    Providers,
    _drain_greeting_era_asr,
    _kick_main_llm_prewarm,
    _reap_main_llm_prewarm,
    _start_listen_pumps,
    _start_monitors,
    _stop_listen_pumps,
)
from isales_common.providers._models import ASRResult
from isales_engine.run_loop import _ListenContext
from tests.test_run_loop import _make_config, _make_providers, _make_session


async def _setup() -> tuple[MockTelephonyClient, CallSession]:
    session = _make_session()
    await session.bus.start()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await tel.dial(session.call_record_id, "+x")  # emits "connected"
    return tel, session


async def _teardown(session: CallSession, ctx: object) -> None:
    await _stop_listen_pumps(session, ctx)  # type: ignore[arg-type]
    assert session.bus is not None
    await session.bus.aclose()


# ---- 2.1 / 2.2: ASR pumps warm during greeting; monitors gated until after --


async def test_start_without_monitors_runs_asr_but_not_monitors() -> None:
    """``start_monitors=False`` starts the ASR + event pumps (connection warms,
    finals flow) but NOT the barge-in monitors — so the greeting cannot be
    barged in on while the connection is being warmed."""
    tel, session = await _setup()
    asr = ScriptedMockASR(partial_step_ms=10)
    ctx = await _start_listen_pumps(
        session, _make_config(), tel, _make_providers(asr), start_monitors=False
    )
    try:
        # ASR + event pumps are up (connection is being warmed).
        assert "asr_pump" in session.tasks
        assert "ev_pump" in session.tasks
        # Monitors are NOT up yet (greeting non-interruptible).
        assert "partial_monitor" not in session.tasks
        assert "vad_monitor" not in session.tasks
        assert ctx.pump_partial_monitor is None
        assert ctx.pump_vad_monitor is None

        # The ASR pump genuinely consumes (proves the connection is live, not
        # merely constructed): a fed turn reaches asr_finals_q.
        await asr.feed_turn("开场白期间客户抢话")
        final = await asyncio.wait_for(ctx.asr_finals_q.get(), timeout=1.0)
        assert final.text == "开场白期间客户抢话"
    finally:
        await _teardown(session, ctx)


async def test_start_monitors_attaches_and_is_idempotent() -> None:
    """After the greeting, :func:`_start_monitors` attaches both monitors; a
    second call is a no-op (does not double-spawn)."""
    tel, session = await _setup()
    ctx = await _start_listen_pumps(
        session,
        _make_config(),
        tel,
        _make_providers(ScriptedMockASR(partial_step_ms=10)),
        start_monitors=False,
    )
    try:
        _start_monitors(session, _make_config(), tel, ctx)
        first_partial = ctx.pump_partial_monitor
        first_vad = ctx.pump_vad_monitor
        assert first_partial is not None
        assert first_vad is not None
        assert "partial_monitor" in session.tasks
        assert "vad_monitor" in session.tasks

        # Idempotent: a second call must not replace the running monitors.
        _start_monitors(session, _make_config(), tel, ctx)
        assert ctx.pump_partial_monitor is first_partial
        assert ctx.pump_vad_monitor is first_vad
    finally:
        await _teardown(session, ctx)


async def test_default_start_monitors_true_keeps_original_behavior() -> None:
    """The original all-at-once path (no pre-warm caller) still starts all four
    pumps together."""
    tel, session = await _setup()
    ctx = await _start_listen_pumps(
        session, _make_config(), tel, _make_providers(ScriptedMockASR(partial_step_ms=10))
    )
    try:
        for key in ("asr_pump", "ev_pump", "partial_monitor", "vad_monitor"):
            assert key in session.tasks
        assert ctx.pump_partial_monitor is not None
        assert ctx.pump_vad_monitor is not None
    finally:
        await _teardown(session, ctx)


# ---- 2.2: greeting-era results are drained --------------------------------


def test_drain_greeting_era_asr_clears_queues_and_anchor() -> None:
    """Greeting-era partials/finals are discarded and the speech-start anchor is
    reset, so greeting-era audio is never consumed as the first user turn."""
    session = _make_session()
    ctx = _ListenContext(
        asr_finals_q=asyncio.Queue(),
        asr_partials_q=asyncio.Queue(),
        hangup_event=asyncio.Event(),
        asr_finished=asyncio.Event(),
        bus=session.bus,
        pump_asr=None,  # type: ignore[arg-type]
        pump_ev=None,  # type: ignore[arg-type]
    )
    ctx.asr_finals_q.put_nowait(ASRResult(text="开场白回声", is_final=True, timestamp_ms=0))
    ctx.asr_partials_q.put_nowait(ASRResult(text="开", is_final=False, timestamp_ms=0))
    session.current_user_speech_started_ms = 12345

    _drain_greeting_era_asr(session, ctx)

    assert ctx.asr_finals_q.empty()
    assert ctx.asr_partials_q.empty()
    assert session.current_user_speech_started_ms is None


# ---- 3.1 / 3.2: main-LLM pre-warm (fixed greeting only) -------------------


async def test_prewarm_fires_chat_for_fixed_greeting() -> None:
    """A fixed-template greeting (no LLM call) fires one lightweight ``chat``
    probe on the main slot to warm its keep-alive pool."""
    session = _make_session()
    config = _make_config()  # fixed_greeting set by default
    assert config.fixed_greeting
    providers = _make_providers()

    task = _kick_main_llm_prewarm(session, config, providers)
    assert task is not None
    await _reap_main_llm_prewarm(task)

    # Exactly one probe chat, carrying a minimal user message.
    assert len(providers.llm.calls) == 1  # type: ignore[attr-defined]
    messages, _json_mode = providers.llm.calls[0]  # type: ignore[attr-defined]
    assert any(m.role == "user" for m in messages)


async def test_no_prewarm_probe_for_llm_greeting() -> None:
    """An LLM greeting warms the same main-slot client via generate_greeting's
    own chat — so the pre-warm must NOT fire a redundant probe."""
    session = _make_session()
    config = replace(_make_config(), fixed_greeting=None)
    providers = _make_providers()

    task = _kick_main_llm_prewarm(session, config, providers)
    assert task is None
    assert providers.llm.calls == []  # type: ignore[attr-defined]


# ---- 3.3: pre-warm fail-soft ----------------------------------------------


class _RaisingLLM(LLMProvider):
    """Main-slot client whose chat always raises (vendor down / bad creds)."""

    async def chat(self, messages, **_kw) -> LLMResponse:  # type: ignore[override]
        raise RuntimeError("vendor unavailable")

    async def chat_stream(self, messages, **_kw):  # type: ignore[override]
        if False:  # pragma: no cover — never streamed in this test
            yield ""
        raise RuntimeError("vendor unavailable")


async def test_prewarm_failure_is_failsoft() -> None:
    """A failing probe must NOT raise out of the pre-warm path — turn-1 falls
    back to a lazy connect."""
    session = _make_session()
    config = _make_config()
    providers = Providers(
        llm=_RaisingLLM(),
        asr=ScriptedMockASR(partial_step_ms=10),
        tts=_make_providers().tts,
    )

    task = _kick_main_llm_prewarm(session, config, providers)
    assert task is not None
    # Reaping must not propagate the probe's RuntimeError.
    await _reap_main_llm_prewarm(task)
    assert task.done()
    assert task.exception() is None
