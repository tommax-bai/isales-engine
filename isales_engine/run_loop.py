"""End-to-end CallSession driver.

Spec: cross-cut — wires state machine + pipeline + filler + transfer + wrap-up
+ silence + telephony + event publishing into a single async loop.

Stage-4 scope notes (read me before extending):

* The "real-time interruption during SPEAKING" path requires monitoring ASR
  partials concurrently with TTS playback; this stage 4 implementation
  performs a *post-speaking* check (the user can interrupt during LISTENING
  but the in-flight TTS is not preempted by mid-speech). Stage-5/6 work
  re-enables the parallel partial monitor by spawning an audio_in→ASR
  consumer task while ``audio_out`` runs.
* Same caveat for silence: we treat LISTENING as a single ``await ASR final``
  with a ``silence_threshold_ms`` timeout; the multi-step
  ACTIVATE → reset → REACTIVATE cycle is folded into the same loop.
* ``no_progress_timer`` is enforced as a global wall-clock check after each
  user turn that is judged "non-progress" (we count user speeches whose
  ASR result is empty / whitelist-only).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from isales_common.enums import CallStatus, HangupCause, TransferStatus
from isales_common.providers._models import ASRResult
from isales_common.providers.asr import ASRProvider
from isales_common.providers.llm import LLMProvider
from isales_common.providers.tts import TTSProvider
from isales_common.schemas.messages.engine_event import (
    ASRFinal,
    CallStarted,
    StatusChanged,
    TranscriptAppended,
)

from isales_engine.call_session import CallSession
from isales_engine.event_publisher import EventPublisher
from isales_engine.pipeline.greeting import generate_greeting
from isales_engine.pipeline.orchestrator import run_pipeline
from isales_engine.realtime.filler_manager import FillerManager
from isales_engine.realtime.interruption_detector import (
    InterruptionConfig,
)
from isales_engine.realtime.no_progress_timer import is_no_progress_exceeded
from isales_engine.realtime.silence_detector import SilenceConfig, evaluate_silence
from isales_engine.realtime.telephony_client import TelephonyClient
from isales_engine.runtime_config import RuntimeConfig
from isales_engine.state_machine import StateMachine
from isales_engine.transcript_recorder import now_utc
from isales_engine.transfer.manager import evaluate_transfer
from isales_engine.wrapup.manager import evaluate_wrap_up

logger = logging.getLogger(__name__)


@dataclass
class Providers:
    llm: LLMProvider
    asr: ASRProvider
    tts: TTSProvider


async def run_session(
    session: CallSession,
    *,
    phone: str,
    config: RuntimeConfig,
    telephony: TelephonyClient,
    providers: Providers,
    publisher: EventPublisher | None = None,
    pipeline_timeout_ms: int = 8000,
    connect_timeout_s: float = 30.0,
) -> None:
    """Drive one call from dial to END. Errors become hangup_cause sentinels."""

    sm = StateMachine(session)
    session.call_started_at_monotonic = time.monotonic()
    started_wall = now_utc()
    if publisher is not None:
        publisher.publish(
            session.campaign_id,
            CallStarted(
                call_record_id=session.call_record_id, started_at=started_wall
            ),
        )

    try:
        connected = await _dial_and_wait_connected(
            session, telephony, phone, timeout_s=connect_timeout_s
        )
        if not connected:
            sm.transition_to(
                CallStatus.END, reason="no_answer", force=True
            )
            session.hangup_cause = HangupCause.NO_ANSWER.value
            session.append_event("hangup", reason="no_answer", initiated_by="ai")
            return
        sm.transition_to(CallStatus.GREETING, reason="connected")
        _publish_status(publisher, session, "connected")

        await _do_greeting(session, config, providers, telephony)
        sm.transition_to(CallStatus.LISTENING, reason="greeting_done")
        _publish_status(publisher, session, "greeting_done")

        await _main_turn_loop(
            session,
            sm,
            config=config,
            telephony=telephony,
            providers=providers,
            publisher=publisher,
            pipeline_timeout_ms=pipeline_timeout_ms,
        )

    except _ManualHangupRequested:
        # API-driven manual hangup (EngineControl).
        await _attempt_hangup(telephony, session.call_record_id)
        if session.state is not CallStatus.END:
            sm.transition_to(CallStatus.END, reason="manual_hangup", force=True)
        session.hangup_cause = HangupCause.MANUAL_HANGUP.value
        session.append_event("hangup", reason="manual_hangup", initiated_by="ai")
    except Exception:
        logger.exception(
            "run_session_unexpected_error call_record_id=%s", session.call_record_id
        )
        if session.state is not CallStatus.END:
            sm.transition_to(
                CallStatus.END, reason="engine_internal_error", force=True
            )
        if session.hangup_cause is None:
            session.hangup_cause = HangupCause.NO_PROGRESS_TIMEOUT.value
        session.append_event(
            "hangup", reason=session.hangup_cause, initiated_by="ai"
        )


# ---- Manual-hangup signal -------------------------------------------------


class _ManualHangupRequested(Exception):
    """Raised inside the run loop when EngineControl requests a hangup."""


def request_manual_hangup(session: CallSession) -> None:
    """API → run-loop bridge: schedule a manual-hangup raise on next await."""

    main = session.tasks.get("main")
    if main is None or main.done():
        return
    # Use a custom marker on the session so the loop can convert the next
    # awaited cancellation into a clean ManualHangupRequested.
    session.hangup_cause = HangupCause.MANUAL_HANGUP.value
    main.cancel()


# ---- Steps ----------------------------------------------------------------


async def _dial_and_wait_connected(
    session: CallSession,
    telephony: TelephonyClient,
    phone: str,
    *,
    timeout_s: float,
) -> bool:
    await telephony.dial(session.call_record_id, phone)
    try:
        async with asyncio.timeout(timeout_s):
            async for ev in telephony.events(session.call_record_id):
                if ev.type == "connected":
                    return True
                if ev.type in ("local_hangup", "remote_hangup", "device_error"):
                    return False
    except TimeoutError:
        return False
    return False


async def _do_greeting(
    session: CallSession,
    config: RuntimeConfig,
    providers: Providers,
    telephony: TelephonyClient,
) -> None:
    text = await generate_greeting(
        session,
        config.pipeline,
        providers.llm,
        fixed_template=config.fixed_greeting,
    )
    session.append_event("greeting", text=text)
    await _play_tts(session, telephony, providers.tts, text, voice_id=config.voice_id)


async def _main_turn_loop(
    session: CallSession,
    sm: StateMachine,
    *,
    config: RuntimeConfig,
    telephony: TelephonyClient,
    providers: Providers,
    publisher: EventPublisher | None,
    pipeline_timeout_ms: int,
) -> None:
    asr_iter = providers.asr.stream_recognize(telephony.audio_in(session.call_record_id))
    events_iter = telephony.events(session.call_record_id)

    asr_finals_q: asyncio.Queue[ASRResult] = asyncio.Queue()
    hangup_event = asyncio.Event()
    asr_finished = asyncio.Event()

    async def _asr_pump() -> None:
        try:
            async for r in asr_iter:
                if r.is_final and r.text:
                    await asr_finals_q.put(r)
        except asyncio.CancelledError:
            pass
        finally:
            asr_finished.set()

    async def _ev_pump() -> None:
        try:
            async for ev in events_iter:
                ev_type = getattr(ev, "type", None)
                if ev_type in ("remote_hangup", "local_hangup", "device_error"):
                    hangup_event.set()
                    return
        except asyncio.CancelledError:
            pass

    pump_asr = asyncio.create_task(_asr_pump(), name="asr_pump")
    pump_ev = asyncio.create_task(_ev_pump(), name="ev_pump")
    session.tasks["asr_pump"] = pump_asr
    session.tasks["ev_pump"] = pump_ev

    no_progress_started = time.monotonic()
    last_progress_ms = _to_ms(no_progress_started)

    try:
        while session.state is not CallStatus.END:
            outcome = await _await_user_or_silence(
                session,
                asr_finals_q=asr_finals_q,
                hangup_event=hangup_event,
                silence_cfg=config.silence,
                interruption_cfg=config.interruption,
            )

            if outcome.kind == "remote_hangup":
                sm.transition_to(CallStatus.END, reason="user_hangup", force=True)
                session.hangup_cause = HangupCause.USER_HANGUP.value
                session.append_event(
                    "hangup", reason="user_hangup", initiated_by="user"
                )
                return

            if outcome.kind == "silence_hangup":
                await _play_tts(
                    session, telephony, providers.tts, outcome.text or "再见。",
                    voice_id=config.voice_id,
                )
                sm.transition_to(
                    CallStatus.END, reason="silence_max_reached", force=True
                )
                session.hangup_cause = HangupCause.SILENCE_MAX_REACHED.value
                session.append_event(
                    "hangup", reason="silence_max_reached", initiated_by="ai"
                )
                return

            if outcome.kind == "silence_activation":
                sm.transition_to(CallStatus.ACTIVATING, reason="silence_threshold")
                session.append_event(
                    "silence_activation",
                    text=outcome.text or "",
                    activation_index=session.silence_activation_count,
                )
                await _play_tts(
                    session, telephony, providers.tts, outcome.text or "",
                    voice_id=config.voice_id,
                )
                session.silence_activation_count += 1
                session.last_tts_end_at = time.monotonic()
                sm.transition_to(CallStatus.LISTENING, reason="activation_done")
                continue

            if outcome.kind != "user_final" or outcome.text is None:
                if is_no_progress_exceeded(
                    last_progress_ts_ms=last_progress_ms,
                    now_ts_ms=_to_ms(time.monotonic()),
                    max_no_progress_seconds=config.max_no_progress_seconds,
                ):
                    sm.transition_to(
                        CallStatus.END, reason="no_progress_timeout", force=True
                    )
                    session.hangup_cause = HangupCause.NO_PROGRESS_TIMEOUT.value
                    session.append_event(
                        "hangup", reason="no_progress_timeout", initiated_by="ai"
                    )
                    return
                continue

            user_text = outcome.text
            session.append_event("user_speech", text=user_text)
            if publisher is not None:
                publisher.publish(
                    session.campaign_id,
                    ASRFinal(
                        call_record_id=session.call_record_id,
                        text=user_text,
                        timestamp_ms=_to_ms(
                            time.monotonic() - session.call_started_at_monotonic
                        ),
                    ),
                )
            last_progress_ms = _to_ms(time.monotonic())

            decision = await evaluate_transfer(
                user_text=user_text,
                turn_count=len(session.dialog_history),
                goal_achieved=False,
                config=config.transfer,
                llm=providers.llm,
            )
            if decision.triggered:
                sm.transition_to(
                    CallStatus.TRANSFERRING, reason=decision.trigger_type or "transfer"
                )
                session.append_event(
                    "transfer_initiated",
                    trigger_type=decision.trigger_type or "",
                    trigger_detail=decision.trigger_detail,
                )
                await _play_tts(
                    session, telephony, providers.tts, decision.phrase,
                    voice_id=config.voice_id,
                )
                session.transfer_status = TransferStatus.MARKED_FOR_HANDOFF.value
                session.transfer_reason = decision.trigger_type
                session.append_event("transfer_marked", handoff_task_id=0)
                await _attempt_hangup(telephony, session.call_record_id)
                sm.transition_to(
                    CallStatus.END, reason="marked_for_handoff", force=True
                )
                session.hangup_cause = HangupCause.MARKED_FOR_HANDOFF.value
                session.append_event(
                    "hangup", reason="marked_for_handoff", initiated_by="ai"
                )
                return

            is_wrap_up = session.state is CallStatus.WRAPPING_UP
            sm.transition_to(CallStatus.PROCESSING, reason="speech_end")
            _publish_status(publisher, session, "speech_end")

            filler: FillerManager | None = None
            if not is_wrap_up:
                filler = FillerManager(
                    session,
                    config.fillers,
                    telephony=telephony,
                    tts=providers.tts,
                    voice_id=config.voice_id,
                )
                await filler.start()

            result = await run_pipeline(
                session,
                user_text,
                config.pipeline,
                providers.llm,
                is_wrap_up=is_wrap_up,
                pipeline_timeout_ms=pipeline_timeout_ms,
            )
            if filler is not None:
                await filler.wait_finished()

            sm.transition_to(CallStatus.SPEAKING, reason="pipeline_done")
            await _play_tts(
                session, telephony, providers.tts, result.reply,
                voice_id=config.voice_id,
            )
            session.append_event(
                "ai_reply",
                text=result.reply,
                turn_id=session.current_turn_id,
                selected_role_config_id=(
                    config.pipeline.roles[
                        result.selected_candidate_index
                    ].role_config_id
                    if 0 <= result.selected_candidate_index < len(config.pipeline.roles)
                    else None
                ),
                goal_achieved=result.goal_achieved,
                goal_type=result.goal_type,
                extracted=result.extracted,
                is_wrap_up=is_wrap_up,
            )

            if result.goal_achieved and not is_wrap_up:
                sm.transition_to(CallStatus.WRAPPING_UP, reason="goal_achieved")
                session.wrap_up_started_at_monotonic = time.monotonic()
                session.wrap_up_started_at_wallclock = now_utc()
                session.append_event(
                    "wrap_up_started",
                    rounds_remaining=config.wrap_up.max_rounds,
                    seconds_remaining=config.wrap_up.max_seconds,
                )
                session.append_event(
                    "goal_achieved",
                    goal_type=result.goal_type,
                    extracted=result.extracted,
                )
                # Stay in WRAPPING_UP for subsequent turns — it serves as the
                # umbrella "listen" state during wrap-up (per goal-achievement
                # spec § 切换到简化管线 + § 当前轮回复正常播放).
                continue

            if is_wrap_up:
                session.wrap_up_round_count += 1
                wrap_decision = evaluate_wrap_up(
                    rounds_so_far=session.wrap_up_round_count,
                    started_at_monotonic=session.wrap_up_started_at_monotonic,
                    config=config.wrap_up,
                )
                if not wrap_decision.proceed:
                    await _play_tts(
                        session, telephony, providers.tts, wrap_decision.closing_phrase,
                        voice_id=config.voice_id,
                    )
                    session.append_event(
                        "wrap_up_completed", reason=wrap_decision.reason
                    )
                    sm.transition_to(
                        CallStatus.END, reason="wrap_up_completed", force=True
                    )
                    session.hangup_cause = HangupCause.WRAP_UP_COMPLETED.value
                    session.append_event(
                        "hangup", reason="wrap_up_completed", initiated_by="ai"
                    )
                    return
                # Continue another wrap-up turn. SPEAKING → WRAPPING_UP transition.
                sm.transition_to(CallStatus.WRAPPING_UP, reason="wrap_up_continue")
                continue

            sm.transition_to(CallStatus.LISTENING, reason="tts_done")
    finally:
        for t in (pump_asr, pump_ev):
            t.cancel()
        await asyncio.gather(pump_asr, pump_ev, return_exceptions=True)
        # remove from session.tasks so SessionManager.cancel_all doesn't double-cancel
        session.tasks.pop("asr_pump", None)
        session.tasks.pop("ev_pump", None)
        _ = asr_finished  # silence unused



# ---- Helpers --------------------------------------------------------------


@dataclass
class _UserAwait:
    # kind ∈ user_final / silence_activation / silence_hangup / remote_hangup / no_progress
    kind: str
    text: str | None = None


async def _await_user_or_silence(
    session: CallSession,
    *,
    asr_finals_q: asyncio.Queue[ASRResult],
    hangup_event: asyncio.Event,
    silence_cfg: SilenceConfig,
    interruption_cfg: InterruptionConfig,
) -> _UserAwait:
    """Wait for user final / remote-hangup / silence-timeout from persistent pumps."""

    if hangup_event.is_set():
        return _UserAwait(kind="remote_hangup")

    listen_started = time.monotonic()
    silence_s = silence_cfg.threshold_ms / 1000.0

    asr_get: asyncio.Task[ASRResult] = asyncio.create_task(asr_finals_q.get())
    hangup_wait: asyncio.Task[bool] = asyncio.create_task(hangup_event.wait())

    try:
        done, _pending = await asyncio.wait(
            {asr_get, hangup_wait}, timeout=silence_s, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        asr_get.cancel()
        hangup_wait.cancel()
        raise

    if hangup_wait in done and hangup_event.is_set():
        asr_get.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseException):
            await asr_get
        return _UserAwait(kind="remote_hangup")

    if asr_get in done:
        result = asr_get.result()
        hangup_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseException):
            await hangup_wait
        if result.text.strip() in interruption_cfg.whitelist:
            return _UserAwait(kind="no_progress")
        session.last_user_speech_end_at = time.monotonic()
        return _UserAwait(kind="user_final", text=result.text)

    # Silence timeout — leave pumps running, just decide.
    asr_get.cancel()
    hangup_wait.cancel()
    with contextlib.suppress(asyncio.CancelledError, BaseException):
        await asr_get
    with contextlib.suppress(asyncio.CancelledError, BaseException):
        await hangup_wait

    elapsed = int((time.monotonic() - listen_started) * 1000)
    decision = evaluate_silence(
        silence_elapsed_ms=elapsed,
        activations_so_far=session.silence_activation_count,
        config=silence_cfg,
    )
    if decision.decision == "activate":
        return _UserAwait(kind="silence_activation", text=decision.phrase)
    if decision.decision == "hangup":
        return _UserAwait(kind="silence_hangup", text=decision.phrase)
    return _UserAwait(kind="no_progress")


async def _play_tts(
    session: CallSession,
    telephony: TelephonyClient,
    tts: TTSProvider,
    text: str,
    *,
    voice_id: str,
) -> None:
    """Stream TTS PCM out via the telephony client. Updates ``last_tts_end_at``."""

    async def chunks() -> AsyncIterator[bytes]:
        async for chunk in tts.synthesize_stream(text, voice_id):
            yield chunk

    await telephony.audio_out(session.call_record_id, chunks())
    session.last_tts_end_at = time.monotonic()


async def _attempt_hangup(telephony: TelephonyClient, call_id: int) -> None:
    try:
        await telephony.hangup(call_id)
    except Exception:  # noqa: BLE001
        logger.exception("telephony_hangup_failed call_id=%s", call_id)


def _publish_status(
    publisher: EventPublisher | None, session: CallSession, reason: str
) -> None:
    if publisher is None:
        return
    publisher.publish(
        session.campaign_id,
        StatusChanged(
            call_record_id=session.call_record_id,
            status=session.state,
            reason=reason,
        ),
    )


def _to_ms(value: float) -> int:
    return int(value * 1000)


def _publish_transcript_appended(
    publisher: EventPublisher | None, session: CallSession, event_type: str
) -> None:
    """Optional: announce that a transcript event was appended."""

    if publisher is None or not session.full_transcript:
        return
    last = session.full_transcript[-1]
    publisher.publish(
        session.campaign_id,
        TranscriptAppended(
            call_record_id=session.call_record_id,
            event_type=event_type,
            ts=int(last.get("ts", 0)),
        ),
    )
