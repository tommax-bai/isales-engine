"""End-to-end CallSession driver.

Spec: cross-cut — wires state machine + pipeline + filler + transfer + wrap-up
+ silence + telephony + event publishing into a single async loop.

Stage-5 (impl-engine-providers PR #6) added:

* **Real-time interruption during SPEAKING / FILLER.** The ASR pump now
  multiplexes partials → ``asr_partials_q`` and finals → ``asr_finals_q``.
  ``_partial_monitor`` consumes partials, runs ``evaluate_partial``, and on
  ``triggered`` cancels ``session.current_speaking_task`` (which is set by
  ``_play_tts``).
* **Continuous-interruption protection** per ai-pipeline spec delta: counter
  ≥ ``max_continuous_interruptions`` triggers ``short_reply`` (prompt-side
  hint to keep the next reply terse) or ``listen_only`` (skip PROCESSING,
  play a short cue, return to LISTENING).
* ``audio_out`` MUST be cancel-aware on every TelephonyClient implementation
  (contract test exercises this).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from isales_common.enums import (
    CallStatus,
    ContinuousInterruptionStrategy,
    HangupCause,
    TransferStatus,
)
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
    evaluate_partial,
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
    token_budget_per_call: int = 50_000,
) -> None:
    """Drive one call from dial to END. Errors become hangup_cause sentinels."""

    # Stash budget on session so the finally block reads it without needing a
    # separate parameter plumbed through every helper.
    session._token_budget_per_call = token_budget_per_call

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

    listen_ctx: _ListenContext | None = None
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

        # Start ASR + partial_monitor BEFORE greeting so the user can barge
        # in mid-greeting. Previously these pumps lived inside _main_turn_loop,
        # which left the greeting structurally uninterruptible.
        listen_ctx = _start_listen_pumps(session, config, telephony, providers)

        await _do_greeting(session, config, providers, telephony)
        # If barge-in fired mid-greeting, partial_monitor flipped
        # ``interruption_signaled``; clear it so the main turn loop's first
        # iteration starts with a clean slate and just picks up the user's
        # ASR final from the queue.
        session.interruption_signaled = False
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
            listen_ctx=listen_ctx,
        )

    except _ManualHangupRequested:
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
    finally:
        if listen_ctx is not None:
            await _stop_listen_pumps(session, listen_ctx)
        # Token-budget warning (impl-engine-providers PR #7). Default 50k
        # if not configured; runtime sets ``_token_budget_per_call`` on the
        # session at construction time.
        budget = getattr(session, "_token_budget_per_call", 50_000)
        if session.total_tokens_in + session.total_tokens_out > budget:
            logger.warning(
                "token_budget_exceeded call_record_id=%s tokens_in=%s "
                "tokens_out=%s budget=%s",
                session.call_record_id,
                session.total_tokens_in,
                session.total_tokens_out,
                budget,
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


@dataclass
class _ListenContext:
    """Shared ASR / event / partial-monitor / VAD state owned by ``run_session``.

    Created by :func:`_start_listen_pumps` BEFORE the greeting so user voice
    during the greeting reaches ``_partial_monitor`` and ``_vad_monitor``
    (both can cancel the in-flight greeting ``_play_tts``). Disposed of by
    :func:`_stop_listen_pumps` in ``run_session``'s ``finally``.
    """

    asr_finals_q: asyncio.Queue[ASRResult]
    asr_partials_q: asyncio.Queue[ASRResult]
    hangup_event: asyncio.Event
    asr_finished: asyncio.Event
    pump_asr: asyncio.Task[None]
    pump_ev: asyncio.Task[None]
    pump_partial_monitor: asyncio.Task[None]
    pump_vad_monitor: asyncio.Task[None]


def _start_listen_pumps(
    session: CallSession,
    config: RuntimeConfig,
    telephony: TelephonyClient,
    providers: Providers,
) -> _ListenContext:
    asr_iter = providers.asr.stream_recognize(
        telephony.audio_in(session.call_record_id)
    )
    events_iter = telephony.events(session.call_record_id)

    asr_finals_q: asyncio.Queue[ASRResult] = asyncio.Queue()
    asr_partials_q: asyncio.Queue[ASRResult] = asyncio.Queue()
    hangup_event = asyncio.Event()
    asr_finished = asyncio.Event()

    async def _asr_pump() -> None:
        try:
            async for r in asr_iter:
                if not r.text:
                    continue
                # TEMP DIAG: classification before the queue split.
                logger.warning(
                    "asr_pump text=%r is_final=%s ts=%s",
                    r.text[:40], r.is_final, r.timestamp_ms,
                )
                if r.is_final:
                    await asr_finals_q.put(r)
                    # A final closes the current utterance — clear the
                    # partial-monitor's speech-start anchor.
                    session.current_user_speech_started_ms = None
                else:
                    await asr_partials_q.put(r)
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
    pump_partial_monitor = asyncio.create_task(
        _partial_monitor(
            session,
            asr_partials_q=asr_partials_q,
            interruption_cfg=config.interruption,
        ),
        name="partial_monitor",
    )
    pump_vad_monitor = asyncio.create_task(
        _vad_monitor(
            session,
            audio_in_vad=telephony.audio_in_vad(session.call_record_id),
            interruption_cfg=config.interruption,
        ),
        name="vad_monitor",
    )
    session.tasks["asr_pump"] = pump_asr
    session.tasks["ev_pump"] = pump_ev
    session.tasks["partial_monitor"] = pump_partial_monitor
    session.tasks["vad_monitor"] = pump_vad_monitor

    return _ListenContext(
        asr_finals_q=asr_finals_q,
        asr_partials_q=asr_partials_q,
        hangup_event=hangup_event,
        asr_finished=asr_finished,
        pump_asr=pump_asr,
        pump_ev=pump_ev,
        pump_partial_monitor=pump_partial_monitor,
        pump_vad_monitor=pump_vad_monitor,
    )


async def _stop_listen_pumps(
    session: CallSession, ctx: _ListenContext
) -> None:
    for t in (
        ctx.pump_asr, ctx.pump_ev, ctx.pump_partial_monitor, ctx.pump_vad_monitor
    ):
        t.cancel()
    await asyncio.gather(
        ctx.pump_asr,
        ctx.pump_ev,
        ctx.pump_partial_monitor,
        ctx.pump_vad_monitor,
        return_exceptions=True,
    )
    session.tasks.pop("asr_pump", None)
    session.tasks.pop("ev_pump", None)
    session.tasks.pop("partial_monitor", None)
    session.tasks.pop("vad_monitor", None)


async def _main_turn_loop(
    session: CallSession,
    sm: StateMachine,
    *,
    config: RuntimeConfig,
    telephony: TelephonyClient,
    providers: Providers,
    publisher: EventPublisher | None,
    pipeline_timeout_ms: int,
    listen_ctx: _ListenContext,
) -> None:
    asr_finals_q = listen_ctx.asr_finals_q
    asr_partials_q = listen_ctx.asr_partials_q
    hangup_event = listen_ctx.hangup_event
    asr_finished = listen_ctx.asr_finished
    pump_asr = listen_ctx.pump_asr
    pump_ev = listen_ctx.pump_ev
    pump_partial_monitor = listen_ctx.pump_partial_monitor

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

            # Continuous-interruption protection (ai-pipeline spec delta).
            protection = _decide_protection(session, config)
            if protection == "listen_only":
                # Skip PROCESSING this turn. Play a short cue + return to
                # LISTENING; counter resets so we give the user a clean slate.
                sm.transition_to(CallStatus.SPEAKING, reason="listen_only_cue")
                await _play_tts(
                    session, telephony, providers.tts,
                    "您请说。",
                    voice_id=config.voice_id,
                    interruptible=False,
                )
                session.consecutive_interruption_count = 0
                config.pipeline.short_reply_active = False
                sm.transition_to(CallStatus.LISTENING, reason="listen_only_done")
                continue
            elif protection == "short_reply":
                config.pipeline.short_reply_active = True
            else:
                config.pipeline.short_reply_active = False

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

            # Token budget bookkeeping (PR #7 wires accumulation; PR #6 just
            # ensures the trace records persist token fields).
            for cand in session.pipeline_trace_records[-1]["role_candidates"]:
                session.total_tokens_in += int(cand.get("prompt_tokens") or 0)
                session.total_tokens_out += int(cand.get("completion_tokens") or 0)

            sm.transition_to(CallStatus.SPEAKING, reason="pipeline_done")
            played = await _play_tts(
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
                interrupted=not played,
            )

            if not played:
                # Real-time interruption fired: SPEAKING got cancelled.
                session.consecutive_interruption_count += 1
                session.interruption_signaled = False
                sm.transition_to(
                    CallStatus.INTERRUPTED, reason="speaking_interrupted"
                )
                # Next loop iteration will await the user's interrupting final.
                continue
            # Successful SPEAKING — clear the counter per spec.
            session.consecutive_interruption_count = 0

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
        # Pump lifecycle now belongs to run_session — these locals just keep
        # type checkers happy that they were "used" in this scope.
        _ = (pump_asr, pump_ev, pump_partial_monitor, asr_finished)



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
    interruptible: bool = True,
) -> bool:
    """Stream TTS PCM out via the telephony client.

    Returns ``True`` on full playback, ``False`` if cancelled mid-stream by
    the partial monitor (real-time interruption). The audio_out implementation
    on every TelephonyClient MUST honour ``CancelledError`` and stop pushing
    chunks (contract test enforces this).

    When ``interruptible=False`` the partial monitor is bypassed for this
    playback (used for protection cues like "您请说" that MUST complete).
    """

    async def chunks() -> AsyncIterator[bytes]:
        async for chunk in tts.synthesize_stream(text, voice_id):
            yield chunk

    play_task = asyncio.create_task(
        telephony.audio_out(session.call_record_id, chunks()), name="audio_out"
    )
    if interruptible:
        session.current_speaking_task = play_task
    try:
        await play_task
    except asyncio.CancelledError:
        if play_task.cancelled():
            session.last_tts_end_at = time.monotonic()
            return False
        raise
    finally:
        if interruptible and session.current_speaking_task is play_task:
            session.current_speaking_task = None

    session.last_tts_end_at = time.monotonic()
    return True


async def _partial_monitor(
    session: CallSession,
    *,
    asr_partials_q: asyncio.Queue[ASRResult],
    interruption_cfg: InterruptionConfig,
) -> None:
    """Long-running task: watch ASR partials during SPEAKING / FILLER.

    On a verdict of ``triggered`` the monitor cancels the in-flight
    ``current_speaking_task`` (if any) and sets
    ``session.interruption_signaled``. Once signaled it stays True until the
    main loop reads + clears it (per spec § "打断判定不可撤销").
    """

    while True:
        try:
            partial = await asr_partials_q.get()
        except asyncio.CancelledError:
            return

        if session.current_speaking_task is None or session.interruption_signaled:
            # Not in SPEAKING / FILLER — partials are just status updates.
            continue

        # Track the speech-start anchor on the first non-empty partial of an
        # utterance so evaluate_partial can apply the duration condition.
        text = partial.text.strip()
        if not text:
            continue

        # Use wall-clock for the duration condition rather than the ASR's
        # ``partial.timestamp_ms``. The V3 SAUC ASR sets ``timestamp_ms`` to
        # the utterance ``end_time`` (audio-domain), which stays constant
        # across repeated confirmation partials of the same word — so
        # ``elapsed = now - anchor`` was permanently 0 and the duration
        # condition never fired (root cause of the "AI plays full sentence
        # even when user speaks throughout" symptom).
        now_ms = int(time.monotonic() * 1000)
        if session.current_user_speech_started_ms is None:
            session.current_user_speech_started_ms = now_ms

        verdict = evaluate_partial(
            text=text,
            speech_started_ts_ms=session.current_user_speech_started_ms,
            now_ts_ms=now_ms,
            config=interruption_cfg,
        )
        # TEMP DIAG: surfaces each evaluation so a follow-up smoke can confirm
        # the verdict path. Remove once verified.
        logger.warning(
            "partial_monitor_eval text=%r elapsed_ms=%s verdict=%s reason=%s",
            text[:40],
            now_ms - session.current_user_speech_started_ms,
            verdict.verdict,
            verdict.reason,
        )
        if verdict.verdict != "triggered":
            continue

        # Lock in the interruption: signal first, then cancel. If cancellation
        # arrives while the play_task hasn't started awaiting yet, the signal
        # makes _play_tts honour it.
        session.interruption_signaled = True
        session.append_event(
            "interruption",
            interrupted_event_id=session.current_turn_id,
            user_text_at_interruption=text,
        )
        speaking_task = session.current_speaking_task
        if speaking_task is not None and not speaking_task.done():
            speaking_task.cancel()


async def _vad_monitor(
    session: CallSession,
    *,
    audio_in_vad: AsyncIterator[bytes],
    interruption_cfg: InterruptionConfig,
) -> None:
    """RMS-based barge-in detector independent of the ASR partial pipeline.

    The partial-monitor path waits for the ASR vendor to emit a transcribed
    partial (V3 SAUC: ~500–800 ms vendor latency) before the duration
    condition can even start counting; that pushes the user-perceived
    cancel latency to ~1.5–2 s. This monitor reads raw PCM directly from
    ``telephony.audio_in_vad()`` and triggers cancel as soon as the rolling
    voice-active span exceeds ``interruption_cfg.min_duration_ms``,
    typically ~250–400 ms after the user actually starts speaking.

    Coordination with ``_partial_monitor``:

    * Both paths gate on ``session.current_speaking_task is not None`` and
      ``not session.interruption_signaled``. Once either fires it sets the
      flag and the other path becomes a no-op for this utterance.
    * The ``interruption`` event emitted here carries ``source="vad"`` so
      transcripts can distinguish the two paths.
    """

    try:
        import audioop  # noqa: PLC0415
    except ImportError:  # pragma: no cover  - Py 3.13+ replacement TBD
        audioop = None  # type: ignore[assignment]

    # 16 kHz int16 mono: a 20 ms frame is 640 bytes. Engine inbound stream is
    # resampled to 16 kHz upstream of audio_in_vad. We tolerate any frame
    # size (push_external_audio is 10 ms = 320 bytes; OnPlaybackAudioFrame
    # is 20–60 ms depending on the SDK build) and infer the duration from
    # the byte count.
    bytes_per_ms = 16_000 * 2 // 1000  # = 32
    # RMS threshold tuned empirically against DingRTC's mixed-playback
    # background-noise floor (~30–80 with no peer speech). Voice utterances
    # typically register ≥ 500–2000. 200 is a safe "above-noise" cutoff
    # that catches normal speech while ignoring DingRTC's mixer hum.
    voice_rms_threshold = 200
    voice_active_ms = 0
    try:
        async for pcm in audio_in_vad:
            if not pcm:
                continue
            frame_ms = max(len(pcm) // bytes_per_ms, 1)
            if audioop is not None:
                try:
                    rms = audioop.rms(pcm, 2)
                except audioop.error:
                    rms = 0
            else:
                rms = 0

            if rms >= voice_rms_threshold:
                voice_active_ms += frame_ms
            else:
                voice_active_ms = 0
                continue

            if session.current_speaking_task is None or session.interruption_signaled:
                # Not in SPEAKING / FILLER, or another path already fired —
                # keep accumulating voice_active_ms so the surfaced count
                # in the event reflects the user's actual span, but skip
                # the cancel side-effect.
                continue

            if voice_active_ms < interruption_cfg.min_duration_ms:
                continue

            # Lock in the interruption: signal first, then cancel.
            session.interruption_signaled = True
            session.append_event(
                "interruption",
                interrupted_event_id=session.current_turn_id,
                source="vad",
                voice_active_ms=voice_active_ms,
                rms=rms,
            )
            speaking_task = session.current_speaking_task
            if speaking_task is not None and not speaking_task.done():
                speaking_task.cancel()
            # Reset so a new utterance after the interrupted reply starts
            # fresh. ``interruption_signaled`` is cleared by the main loop
            # after it processes the interruption.
            voice_active_ms = 0
    except asyncio.CancelledError:
        return


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


def _decide_protection(
    session: CallSession, config: RuntimeConfig
) -> str | None:
    """Return ``'short_reply'`` / ``'listen_only'`` / ``None`` per ai-pipeline spec."""

    if session.consecutive_interruption_count < config._max_continuous_interruptions:
        return None
    if (
        config._continuous_interruption_strategy
        == ContinuousInterruptionStrategy.LISTEN_ONLY.value
    ):
        return "listen_only"
    return "short_reply"


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
