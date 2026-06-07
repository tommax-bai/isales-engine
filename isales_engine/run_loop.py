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
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

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
from isales_common.schemas.pipeline import MainSpec

from isales_engine.call_session import CallSession
from isales_engine.event_publisher import EventPublisher
from isales_engine.eventbus import EventBus, Lane
from isales_engine.events import (
    AsrFinal,
    AsrPartial,
    AsrStreamEnded,
    HangupDetected,
    ManualHangupRequested,
    TransferRequested,
)
from isales_engine.pipeline.decider import DeciderAction, decide
from isales_engine.pipeline.greeting import generate_greeting
from isales_engine.pipeline.orchestrator import (
    PipelineStream,
    run_pipeline_stream,
    run_restructure_stream,
)
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
from isales_engine.streaming.types import RefereeResult
from isales_engine.transcript_recorder import now_utc
from isales_engine.transfer.manager import (
    TransferDecision,
    evaluate_transfer_cheap,
    evaluate_transfer_llm,
)
from isales_engine.wrapup.manager import evaluate_wrap_up

logger = logging.getLogger(__name__)


class Directive(Enum):
    """What the turn loop does after a gate-first turn runs.

    * ``CONTINUE`` — next loop iteration (await the next user final / silence).
    * ``RETURN``   — end the call (terminal route / hangup / transfer).
    """

    CONTINUE = "continue"
    RETURN = "return"


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

    # Stash extractor slot ids so finalize_session can LPUSH the post-call
    # extract task without re-loading the runtime config (pipeline-stream-and-
    # referee § post-call extractor).
    session.extractor_role_config_id = config.pipeline.extractor.role_config_id
    session.extractor_prompt_version_id = config.pipeline.extractor.prompt_version_id

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
    bus = session.bus  # constructed WITH the session (CallSession.bus factory)
    try:
        # The per-call EventBus is created with the session, BEFORE dial_consumer
        # registers it / sets tasks["main"]. So a control-plane hangup/transfer
        # (Redis → _on_manual_hangup → bus.post) is never dropped: an event that
        # lands in the pre-dial runner preamble — the DB load that runs before
        # run_session even starts — is buffered on the lossless lane and handled
        # once we start() the dispatchers below. This is byte-identical to the
        # old request_manual_hangup() reach, which fired the instant
        # dial_consumer set tasks["main"]. (Earlier the bus was created HERE,
        # which left that preamble window uncovered — a real, now-closed gap.)
        # Here we subscribe the control bridge + start the dispatchers;
        # _start_listen_pumps adds the ASR producers/bridges after the greeting.
        _subscribe_control_bridges(session, bus)
        await bus.start()

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
        sm.transition_to(CallStatus.IN_CALL, reason="connected")
        _publish_status(publisher, session, "connected")

        # NOTE on greeting interruptibility: listen pumps start AFTER the
        # greeting, so the greeting cannot be barged in on. (An earlier
        # 2026-05-28 rationale attributing mid-greeting false-triggers to a
        # DingRTC POSITION_PLAYBACK self-loopback / mixed-playback echo was
        # retracted 2026-06-04 — call 138 evidence does not support it. If
        # mid-greeting barge-in is revisited, diagnose against the real ASR
        # connection lifecycle first; do not reintroduce the loopback/AEC
        # theory without fresh evidence.)

        await _do_greeting(session, config, providers, telephony)
        sm.transition_to(CallStatus.IN_CALL, reason="greeting_done")
        _publish_status(publisher, session, "greeting_done")

        listen_ctx = await _start_listen_pumps(session, config, telephony, providers)

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
        # bus is always present (constructed with the session); pumps are now
        # cancelled (their _asr_pump finally already posted the final
        # AsrStreamEnded onto the still-open bus), so aclose() drains the
        # lossless lane then stops both dispatchers. Surface the one
        # intentionally-lossy deviation: AsrPartial rides the bounded(256)
        # drop-oldest lane, so a dropped partial is otherwise silent.
        await bus.aclose()
        if bus.dropped:
            logger.warning(
                "eventbus_lossy_dropped call_record_id=%s dropped=%s — "
                "AsrPartial lane overflowed (maxsize=256); unexpected at "
                "human ASR cadence",
                session.call_record_id,
                bus.dropped,
            )
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


# ---- Control-plane bridges (manual hangup / transfer) ---------------------


def _subscribe_control_bridges(session: CallSession, bus: EventBus) -> None:
    """Wire the Redis control-plane signals onto the per-call bus.

    main.py's ``_on_manual_hangup`` / ``_on_transfer`` (driven by the
    EngineControl pub/sub consumer) ``bus.post`` a ``ManualHangupRequested`` /
    ``TransferRequested``; this terminates the call by cancelling the run-loop
    ``main`` task — byte-identical to the deleted ``request_manual_hangup()``
    ``main.cancel()`` sentinel, minus the never-raised ``_ManualHangupRequested``
    exception (the old ``except`` clause was dead code: ``main.cancel()`` only
    ever raised a bare ``CancelledError``). ``hangup_cause`` is set before the
    cancel so the run_session ``finally`` / finalize records it; the cancel
    unwinds run_session (stop pumps + aclose bus).

    Transfer currently routes through the same hangup path — the stage-4 stub
    where ``_on_transfer`` always set MANUAL_HANGUP. Real transfer routing is a
    later change; kept byte-identical here.
    """

    def _terminate(cause: str) -> None:
        main = session.tasks.get("main")
        if main is None or main.done():
            return
        session.hangup_cause = cause
        main.cancel()

    bus.subscribe(
        ManualHangupRequested,
        lambda _ev: _terminate(HangupCause.MANUAL_HANGUP.value),
    )
    bus.subscribe(
        TransferRequested,
        lambda _ev: _terminate(HangupCause.MANUAL_HANGUP.value),
    )


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

    change-1 Phase 1 (engine-eventbus-foundation): the ASR / event pumps now
    PRODUCE typed signals onto ``bus`` and a thin bridge subscriber feeds the
    queues / events below, so the ``_main_turn_loop`` consumer (and the two
    monitors, which still read ``asr_partials_q`` directly) stay byte-identical
    (design.md D4). The queues/events ARE the bridge's downstream sink.
    """

    asr_finals_q: asyncio.Queue[ASRResult]
    asr_partials_q: asyncio.Queue[ASRResult]
    hangup_event: asyncio.Event
    asr_finished: asyncio.Event
    bus: EventBus
    pump_asr: asyncio.Task[None]
    pump_ev: asyncio.Task[None]
    pump_partial_monitor: asyncio.Task[None]
    pump_vad_monitor: asyncio.Task[None]


async def _start_listen_pumps(
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

    # change-1 (engine-eventbus-foundation §2): the ASR / event pumps become bus
    # PRODUCERS; the bridge subscribers below feed the existing queues / events
    # so the _main_turn_loop consumer is byte-identical (D4). The bus itself is
    # created + started + owned by run_session (so the control-plane hangup
    # bridge is live from dial onward); here we only add the ASR producers /
    # bridges to it. AsrPartial is the only droppable signal (registered LOSSY);
    # finals / hangup / stream-end default LOSSLESS — a user turn must never be
    # dropped. (VadFrame's LOSSY registration arrives with the _vad_monitor
    # producer in the later barge-in inversion; nothing posts VadFrame yet.)
    bus = session.bus  # constructed with the session; run_session already start()ed it
    bus.register_lane(AsrPartial, Lane.LOSSY)

    def _bridge_final(ev: AsrFinal) -> None:
        # Both finals consumers (_await_user_or_silence, _partial_monitor) read
        # only ``.text``; timestamp_ms is never read off a queued final, so a
        # minimal reconstruction is byte-identical for the consumer.
        asr_finals_q.put_nowait(
            ASRResult(text=ev.text, is_final=True, timestamp_ms=0)
        )

    def _bridge_partial(ev: AsrPartial) -> None:
        asr_partials_q.put_nowait(
            ASRResult(text=ev.text, is_final=False, timestamp_ms=0)
        )

    def _bridge_hangup(_ev: HangupDetected) -> None:
        hangup_event.set()

    def _bridge_stream_ended(_ev: AsrStreamEnded) -> None:
        asr_finished.set()

    bus.subscribe(AsrFinal, _bridge_final)
    bus.subscribe(AsrPartial, _bridge_partial)
    bus.subscribe(HangupDetected, _bridge_hangup)
    bus.subscribe(AsrStreamEnded, _bridge_stream_ended)
    # Subscribers added before the pumps are spawned, so no posted event can be
    # dispatched before its bridge handler exists. The bus was already started
    # by run_session, which also owns bus.aclose() in its finally — so even if
    # this function raises before returning a context, the bus is still torn
    # down (the earlier change-2 dispatcher-leak hazard is resolved by that
    # ownership move).

    async def _asr_pump() -> None:
        try:
            async for r in asr_iter:
                if not r.text:
                    continue
                if r.is_final:
                    bus.post(AsrFinal(text=r.text))
                    # A final closes the current utterance — clear the
                    # partial-monitor's speech-start anchor.
                    session.current_user_speech_started_ms = None
                else:
                    bus.post(AsrPartial(text=r.text))
        except asyncio.CancelledError:
            pass
        finally:
            bus.post(AsrStreamEnded())

    async def _ev_pump() -> None:
        try:
            async for ev in events_iter:
                ev_type = getattr(ev, "type", None)
                if ev_type in ("remote_hangup", "local_hangup", "device_error"):
                    bus.post(HangupDetected(cause=ev_type))
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
        bus=bus,
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
    # The bus is NOT closed here — it outlives the listen pumps (run_session
    # owns it and aclose()s it in its finally). The producers are now cancelled,
    # so the _asr_pump finally's AsrStreamEnded post lands on the still-open bus
    # and is drained by run_session's aclose().
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
                sm.transition_to(CallStatus.IN_CALL, reason="silence_threshold")
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
                sm.transition_to(CallStatus.IN_CALL, reason="activation_done")
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

            # Cheap (keyword / round) transfer triggers stay inline — zero LLM,
            # deterministic (pipeline-latency-tail § D). The LLM-backed triggers
            # are evaluated off the hot path: preferred via the referee's
            # ``transfer`` decision, with a parallel detector spawned below for
            # campaigns that explicitly configure transfer_intent / transfer_llm.
            decision = evaluate_transfer_cheap(
                user_text=user_text,
                turn_count=len(session.dialog_history),
                goal_achieved=False,
                config=config.transfer,
            )
            if decision.triggered:
                await _perform_handoff(
                    session, sm, telephony, providers.tts,
                    trigger_type=decision.trigger_type or "transfer",
                    trigger_detail=decision.trigger_detail,
                    phrase=decision.phrase,
                    voice_id=config.voice_id,
                )
                return

            is_wrap_up = session.in_wrap_up

            # Continuous-interruption protection (ai-pipeline spec delta).
            protection = _decide_protection(session, config)
            if protection == "listen_only":
                # Skip PROCESSING this turn. Play a short cue + return to
                # LISTENING; counter resets so we give the user a clean slate.
                sm.transition_to(CallStatus.IN_CALL, reason="listen_only_cue")
                await _play_tts(
                    session, telephony, providers.tts,
                    "您请说。",
                    voice_id=config.voice_id,
                    interruptible=False,
                )
                session.consecutive_interruption_count = 0
                config.pipeline.short_reply_active = False
                sm.transition_to(CallStatus.IN_CALL, reason="listen_only_done")
                continue
            elif protection == "short_reply":
                config.pipeline.short_reply_active = True
            else:
                config.pipeline.short_reply_active = False

            sm.transition_to(CallStatus.IN_CALL, reason="speech_end")
            _publish_status(publisher, session, "speech_end")
            ts_start = now_utc()
            processing_start = time.monotonic()

            # Transfer LLM detection (pipeline-latency-tail § D): for campaigns
            # that explicitly configure transfer_intent / transfer_llm, spawn
            # the LLM detector in parallel with main streaming so it never
            # blocks the PROCESSING entry. Resolved before this SPEAKING turn
            # finalizes (alongside the referee). Off → None, zero cost.
            transfer_llm_task: asyncio.Task[TransferDecision] | None = None
            if (
                not is_wrap_up
                and (config.transfer.intent_enabled or config.transfer.llm_enabled)
            ):
                transfer_llm_task = asyncio.create_task(
                    evaluate_transfer_llm(
                        user_text=user_text,
                        config=config.transfer,
                        llm=providers.llm,
                    ),
                    name="transfer_llm",
                )

            # prompt_versions snapshot: mark the wrap-up append flag for this
            # call (role-prompt § prompt_versions 快照标记).
            if is_wrap_up and isinstance(session.prompt_versions_snapshot, dict):
                session.prompt_versions_snapshot["wrap_up_appended"] = True

            # FILLER is off by default (streaming reaches first audio ~500ms).
            # When a campaign opts in, filler overlaps the main stream and is
            # preempted by the first main sentence — no blocking gate.
            # Time-gated filler (tts-cache-and-gated-filler § B): build the
            # manager but do NOT start it here — _play_streaming starts it only
            # if the main reply's first audio hasn't begun within
            # ``filler_delay_ms``. (Cached filler audio masks a slow LLM TTFT
            # without polluting fast turns.)
            filler: FillerManager | None = None
            if config.filler_enabled and not is_wrap_up:
                filler = FillerManager(
                    session,
                    config.fillers,
                    telephony=telephony,
                    tts=providers.tts,
                    voice_id=config.voice_id,
                )

            # engine-tools-multidialogue-gating: gate-first turn — spawn
            # referees + eager dialogue candidates (main + opt-in personas),
            # await the pre-reply gate, then release ONE route (or a lazy tool).
            # Phase-4 deleted the legacy speak-then-judge body + the
            # ENGINE_USE_ROUTER kill-switch + the change-2 effect-route adapter;
            # this is the sole turn path now.
            directive = await _run_gated_turn(
                session,
                sm,
                config=config,
                telephony=telephony,
                providers=providers,
                publisher=publisher,
                user_text=user_text,
                ts_start=ts_start,
                processing_start=processing_start,
                is_wrap_up=is_wrap_up,
                filler=filler,
                transfer_llm_task=transfer_llm_task,
                pipeline_timeout_ms=pipeline_timeout_ms,
            )
            if directive is Directive.RETURN:
                return
            continue
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
        # Coalesce any finals that queued while the AI was busy speaking: when
        # the user out-paces the AI (or talks through a long reply that didn't
        # barge in), multiple finals pile up in asr_finals_q while the main
        # loop is blocked in playback. Responding to each one turn-by-turn is
        # the "憋住 then flood" symptom — drain the backlog and answer ONCE to
        # the combined input (the user's most recent intent), instead of
        # replaying the whole queue.
        texts = [result.text.strip()]
        while True:
            try:
                extra = asr_finals_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if extra.text.strip():
                texts.append(extra.text.strip())
        combined = " ".join(t for t in texts if t)
        if not combined or combined in interruption_cfg.whitelist:
            return _UserAwait(kind="no_progress")
        session.last_user_speech_end_at = time.monotonic()
        return _UserAwait(kind="user_final", text=combined)

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


class _SynthJob:
    """A pre-synthesizing TTS job (pipeline-latency-tail § B).

    Construction immediately spawns a background task that drains
    ``tts.synthesize_stream(text)`` into an internal buffer — so the vendor
    is already producing audio for sentence N+1 while the consumer plays
    sentence N. ``stream()`` replays the buffered PCM (and re-raises any
    synthesis error after draining). ``aclose()`` cancels the in-flight
    synthesis (barge-in / teardown) so we don't leak a vendor connection.
    """

    def __init__(self, text: str, tts: TTSProvider, voice_id: str) -> None:
        self.text = text
        self._buffer: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._error: BaseException | None = None
        # Own the TTS iterator so aclose() can close it even when the drain
        # task's own cleanup gets cut short by cancellation.
        self._gen = tts.synthesize_stream(text, voice_id)
        self._task = asyncio.create_task(self._drain(), name="tts_presynth")

    async def _drain(self) -> None:
        try:
            async for chunk in self._gen:
                await self._buffer.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced via stream()
            self._error = exc
        finally:
            await self._buffer.put(None)  # sentinel: synthesis done

    async def stream(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._buffer.get()
            if chunk is None:
                break
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        # Explicitly close the TTS iterator so its vendor connection / SSE
        # stream is released NOW (on barge-in / teardown). The drain task's
        # own cleanup can be cut short by its cancellation, so we close the
        # generator here — _play_streaming's finally runs uncancelled.
        aclose = getattr(self._gen, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


async def _play_streaming(
    session: CallSession,
    telephony: TelephonyClient,
    tts: TTSProvider,
    stream: PipelineStream,
    *,
    voice_id: str,
    filler: FillerManager | None,
    filler_delay_s: float = 0.6,
    processing_start: float,
) -> tuple[int | None, bool]:
    """Play the main LLM streaming reply via a producer/consumer pipeline.

    Producer consumes ``chat_stream → split_sentences`` and, for each
    sentence, **immediately** kicks off TTS synthesis (``_SynthJob``),
    pushing the ready job onto a bounded queue (``maxsize=2``). The consumer
    pulls jobs and plays them — so while sentence N plays, sentence N+1 is
    already synthesizing in the vendor. This kills the per-sentence TTS
    first-byte silence gap and stops the main LLM from being suspended by
    playback (pipeline-latency-tail § B).

    Returns ``(first_audio_ms, fully_played)``. ``first_audio_ms`` is measured
    from PROCESSING entry to the first PCM chunk the consumer actually plays.
    On a real-time interruption (partial monitor cancels the audio_out task)
    returns ``(first_audio_ms, False)``; the producer, the main stream
    generator, and every queued/in-flight synthesis are cancelled in the
    ``finally`` so ``chat_stream`` and the vendor TTS connections stop.
    """
    first_audio_ms: int | None = None
    fully_played = True
    sentences = stream.sentences()
    # maxsize=2: producer leads the consumer by at most 2 sentences — enough
    # to hide TTS first-byte latency, bounded so a fast LLM doesn't pre-pay
    # synthesis for many sentences that a barge-in would discard.
    job_q: asyncio.Queue[_SynthJob | None] = asyncio.Queue(maxsize=2)

    async def _producer() -> None:
        pending: _SynthJob | None = None
        try:
            async for sentence in sentences:
                pending = _SynthJob(sentence, tts, voice_id)
                await job_q.put(pending)
                pending = None
        except asyncio.CancelledError:
            # Consumer's finally drains + closes queued jobs; just clean up
            # the one job we created but never enqueued. No sentinel needed.
            if pending is not None:
                await pending.aclose()
            raise
        except Exception as exc:  # noqa: BLE001
            # Sentence generator failed mid-stream (rare — sentences() catches
            # its own LLM errors). Stop producing but still drop the sentinel
            # below so the consumer drains what's queued instead of hanging.
            logger.warning(
                "tts_producer_stream_error turn=%s: %s", stream.turn_id, exc
            )
            if stream.result.error is None:
                stream.result.error = f"main_stream_producer: {exc}"
        await job_q.put(None)  # completion sentinel (normal or handled-error)

    producer_task = asyncio.create_task(_producer(), name="tts_producer")

    # Time-gated filler (tts-cache-and-gated-filler § B): only play a filler if
    # the first real audio hasn't begun within ``filler_delay_s``. Fast turns
    # never hear it; slow turns get a cached (zero-synth) filler masking the
    # LLM TTFT. Cancelled + stopped the moment the first job plays.
    filler_task: asyncio.Task[None] | None = None
    if filler is not None:
        async def _maybe_filler() -> None:
            await asyncio.sleep(filler_delay_s)
            if first_audio_ms is None:
                await filler.start()
        filler_task = asyncio.create_task(_maybe_filler(), name="filler_gate")

    current_job: _SynthJob | None = None
    filler_stopped = False
    try:
        while True:
            next_job = await job_q.get()
            if next_job is None:
                break
            job: _SynthJob = next_job  # narrowed non-Optional for the closure
            current_job = job
            if not filler_stopped and filler is not None:
                # First real audio is ready: cancel the pending filler timer
                # and preempt any filler already playing (same cancel path as
                # 5/28 barge-in).
                if filler_task is not None:
                    filler_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await filler_task
                await filler.stop()
                filler_stopped = True

            async def _marked(j: _SynthJob = job) -> AsyncIterator[bytes]:
                nonlocal first_audio_ms
                async for chunk in j.stream():
                    if first_audio_ms is None:
                        first_audio_ms = int(
                            (time.monotonic() - processing_start) * 1000
                        )
                    yield chunk

            try:
                played = await _play_chunks(
                    session, telephony, _marked(), interruptible=True
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # TTS synthesis / playback error for this sentence: log + stop
                # this turn's playback without crashing the call (consumer 不
                # 卡死 + 落 error). Remaining queued jobs are closed in finally.
                logger.warning(
                    "tts_playback_error turn=%s text=%r: %s",
                    stream.turn_id, job.text, exc,
                )
                if stream.result.error is None:
                    stream.result.error = f"tts_playback: {exc}"
                break
            if not played:
                # Barge-in: keep current_job set so the finally closes its
                # (possibly still-synthesizing) TTS iterator.
                fully_played = False
                break
            current_job = None  # fully played — nothing to clean up
    finally:
        # Barge-in remainder capture (engine-multi-referee-and-restructure D5 b):
        # on a real interruption of the *main* reply, the not-yet-spoken sentence
        # buffer (the interrupted sentence + still-queued sentences) is preserved
        # so a next-turn restructure rule (source=interrupt_remaining) can
        # re-voice it. Restructure turns don't re-capture (would loop). This does
        # not change the discard timing / cancellation semantics below.
        capture = not fully_played and not stream.is_restructure
        captured: list[str] = []
        if capture and current_job is not None:
            captured.append(current_job.text)
        if filler_task is not None:
            filler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await filler_task
        producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await producer_task
        await sentences.aclose()
        if current_job is not None:
            await current_job.aclose()
        # Drain + close any jobs still queued so their pre-synth TTS iterators
        # are released (avoid leaking vendor connections on barge-in).
        while not job_q.empty():
            pending_job = job_q.get_nowait()
            if pending_job is not None:
                if capture:
                    captured.append(pending_job.text)
                await pending_job.aclose()
        # Eager fan-out (engine-tools-multidialogue-gating): the speculative
        # buffer can lead the bounded job queue, so the not-yet-replayed tail is
        # also unspoken — fold it into the barge-in remainder. Then cancel the
        # eager task so its chat_stream / TTS connections are released. Both
        # no-op for the legacy non-eager path (flag-OFF stays byte-identical).
        if capture and stream.is_eager:
            captured.append(stream.buffer_remainder())
        await stream.cancel_eager()
        if filler is not None:
            await filler.stop()
        if capture:
            remaining = "".join(captured).strip()
            session.interrupt_remaining_text = remaining or None
    return first_audio_ms, fully_played


async def _perform_handoff(
    session: CallSession,
    sm: StateMachine,
    telephony: TelephonyClient,
    tts: TTSProvider,
    *,
    trigger_type: str,
    trigger_detail: str,
    phrase: str,
    voice_id: str,
) -> None:
    """Run the TRANSFERRING → marked-for-handoff → hangup → END flow.

    Shared by the inline cheap triggers, the parallel transfer-LLM detector,
    and the referee transfer decision (pipeline-latency-tail § D). ``phrase``
    is played only when non-empty (the referee path plays nothing — the main
    reply already went out).
    """
    sm.transition_to(CallStatus.TRANSFERRING, reason=trigger_type or "transfer")
    session.append_event(
        "transfer_initiated",
        trigger_type=trigger_type or "",
        trigger_detail=trigger_detail,
    )
    if phrase:
        await _play_tts(session, telephony, tts, phrase, voice_id=voice_id)
    session.transfer_status = TransferStatus.MARKED_FOR_HANDOFF.value
    session.transfer_reason = trigger_type
    session.append_event("transfer_marked", handoff_task_id=0)
    await _attempt_hangup(telephony, session.call_record_id)
    sm.transition_to(CallStatus.END, reason="marked_for_handoff", force=True)
    session.hangup_cause = HangupCause.MARKED_FOR_HANDOFF.value
    session.append_event("hangup", reason="marked_for_handoff", initiated_by="ai")


async def _resolve_transfer_llm(
    task: asyncio.Task[TransferDecision] | None,
) -> TransferDecision | None:
    """Await the parallel transfer-LLM detector with a fail-open budget.

    Mirrors ``_await_referees``: ``None`` when no detector was spawned; on
    timeout / failure the task is cancelled and ``None`` returned (transfer
    never blocks or crashes the turn).
    """
    if task is None:
        return None
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
    except TimeoutError:
        task.cancel()
        return None
    except Exception:  # noqa: BLE001 — transfer detection never blocks the call
        return None


async def _await_referees(
    stream: PipelineStream, *, timeout_s: float = 2.0
) -> list[RefereeResult]:
    """Await all N referee tasks with a per-referee fail-open budget.

    Returns one :class:`RefereeResult` per referee (empty when none were
    spawned — wrap-up / restructure). A timed-out / errored referee yields a
    labelled fail-open result so it simply matches no routing rule; one slow
    referee never blocks the others (they are awaited concurrently).

    ``timeout_s`` defaults to 2.0 (legacy post-reply await, byte-identical). The
    gate-first path passes ``campaign.referee_timeout_ms`` (~600ms) so the
    pre-reply gate fails open fast (engine-tools-multidialogue-gating).
    """
    if not stream.referee_tasks:
        return []

    async def _await_one(task: asyncio.Task[RefereeResult], label: str) -> RefereeResult:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except TimeoutError:
            task.cancel()
            return RefereeResult.fail_open(label=label, reason="timeout")
        except Exception:  # noqa: BLE001 — a referee failure never blocks the call
            return RefereeResult.fail_open(label=label, reason="invalid")

    return list(
        await asyncio.gather(
            *(
                _await_one(t, label)
                for t, label in zip(stream.referee_tasks, stream.referee_labels, strict=False)
            )
        )
    )


# ---- Gate-first turn (engine-tools-multidialogue-gating)
#
# Inverts the legacy speak-then-judge into judge-then-speak: spawn referees +
# eager-buffer the candidate dialogue routes (main + opt-in personas) so
# generation overlaps the gate, AWAIT the referee gate (referee_timeout_ms,
# fail-open to referee_fail_open_route), then decide() -> select ONE route ->
# release the winning buffered dialogue reply (cancel the losers) or execute a
# lazy tool. The selected route's then_state is projected via the StateMachine
# (sole state writer); routes never transition directly. The legacy flag-OFF
# path is byte-identical and untouched.


@dataclass
class _GatedSelection:
    route_id: str
    kind: str  # "dialogue" | "tool"
    then_state: str | None = None
    goal_type: str = ""
    source: str | None = None
    restructure_trigger: str | None = None
    tool_type: str | None = None  # "hangup" | "transfer"
    tool_config: dict[str, Any] | None = None


def _select_gated_route(action: DeciderAction, config: RuntimeConfig) -> _GatedSelection:
    """Map a decider verdict to ONE route for the pre-reply gate.

    ``decide()`` is reused verbatim (first-match-wins); only the action→route
    mapping widens here. Legacy transition/restructure actions are shimmed to
    route + then_state so they flow through the same projection. Unknown personas
    / tool aliases fail open to ``referee_fail_open_route`` (default ``main``) —
    a misconfig never crashes the call.
    """
    fail_open = config.pipeline.referee_fail_open_route or "main"
    tools = config.pipeline.tools

    def _tool(alias: str, then_state: str | None) -> _GatedSelection:
        tc = tools.get(alias)
        if not isinstance(tc, dict) or tc.get("type") not in ("hangup", "transfer"):
            return _GatedSelection(route_id=fail_open, kind="dialogue")
        ttype = tc["type"]
        return _GatedSelection(
            route_id=f"tool:{alias}",
            kind="tool",
            then_state=then_state or ("END" if ttype == "hangup" else "TRANSFERRING"),
            tool_type=ttype,
            tool_config=tc,
        )

    _BUILTIN_THEN = {"closing": "WRAPPING_UP", "recovery": "ACTIVATING", "restructure": "LISTENING"}

    if action.kind == "transition":
        if action.to == "goal_achieved":
            return _GatedSelection(
                route_id="closing", kind="dialogue",
                then_state="WRAPPING_UP", goal_type=action.goal_type or "",
            )
        if action.to == "transfer":
            return _GatedSelection(
                route_id="tool:transfer", kind="tool",
                then_state="TRANSFERRING", tool_type="transfer",
            )
        if action.to == "customer_decline":
            return _GatedSelection(route_id="recovery", kind="dialogue", then_state="ACTIVATING")
    elif action.kind == "restructure":
        return _GatedSelection(
            route_id="restructure", kind="dialogue", then_state="LISTENING",
            source=action.source, restructure_trigger=action.restructure_trigger,
        )
    elif action.kind == "route":
        to = action.to or ""
        if to in _BUILTIN_THEN:
            return _GatedSelection(
                route_id=to, kind="dialogue",
                then_state=action.then_state or _BUILTIN_THEN[to],
                goal_type=action.goal_type or "",
            )
        if to in {p.label for p in config.pipeline.personas}:
            return _GatedSelection(
                route_id=f"persona:{to}", kind="dialogue", then_state=action.then_state
            )
        return _GatedSelection(route_id=fail_open, kind="dialogue")
    elif action.kind == "tool":
        return _tool(action.tool or "", action.then_state)

    # continue / no rule matched → fail open to the main dialogue route
    return _GatedSelection(route_id=fail_open, kind="dialogue")


def _persona_main_spec(persona: Any) -> MainSpec:
    """Re-cast a persona slot as a main slot so the existing prompt builder /
    PipelineStream drive it (the persona's prompt replaces the main prompt)."""
    return MainSpec(
        role_config_id=persona.role_config_id,
        prompt_version_id=persona.prompt_version_id,
        system_prompt=persona.system_prompt,
        model=persona.model,
        temperature=persona.temperature,
        top_p=persona.top_p,
    )


async def _run_gated_turn(
    session: CallSession,
    sm: StateMachine,
    *,
    config: RuntimeConfig,
    telephony: TelephonyClient,
    providers: Providers,
    publisher: EventPublisher | None,
    user_text: str,
    ts_start: object,
    processing_start: float,
    is_wrap_up: bool,
    filler: FillerManager | None,
    transfer_llm_task: asyncio.Task[TransferDecision] | None,
    pipeline_timeout_ms: int,
) -> Directive:
    """Drive ONE gate-first PROCESSING turn. Returns a Directive (RETURN ends
    the call; CONTINUE loops to the next user-await)."""
    voice_id = config.voice_id

    if is_wrap_up:
        return await _gated_wrap_up_turn(
            session, sm, config=config, telephony=telephony, providers=providers,
            user_text=user_text, ts_start=ts_start, processing_start=processing_start,
            pipeline_timeout_ms=pipeline_timeout_ms,
        )

    # ---- eager candidates: main (carries the referees) + opt-in personas ----
    cap = max(1, min(3, config.pipeline.persona_fanout_cap))
    enabled_personas = config.pipeline.personas[: max(0, cap - 1)]
    main_stream = run_pipeline_stream(
        session, user_text, config.pipeline, providers.llm, providers.llm,
        is_wrap_up=False, pipeline_timeout_ms=pipeline_timeout_ms,
    )
    main_stream.start_eager()
    candidates: dict[str, PipelineStream] = {"main": main_stream}
    persona_specs: dict[str, Any] = {}
    for persona in enabled_personas:
        persona_cfg = replace(
            config.pipeline, main=_persona_main_spec(persona), referees=[]
        )
        ps = PipelineStream(
            session, user_text, persona_cfg, providers.llm, providers.llm,
            is_wrap_up=False, pipeline_timeout_ms=pipeline_timeout_ms,
        )
        ps.turn_id = main_stream.turn_id  # one turn id; no extra referees/increment
        ps.start_eager()
        rid = f"persona:{persona.label}"
        candidates[rid] = ps
        persona_specs[rid] = persona
    persona_candidates = list(candidates)

    # ---- GATE: await the referees before releasing any audio ----
    referee_results = await _await_referees(
        main_stream, timeout_s=config.pipeline.referee_timeout_ms / 1000.0
    )
    action: DeciderAction = decide(
        referee_results,
        config.pipeline.routing_rules,
        primary_referee_label=config.pipeline.primary_referee_label,
        restructure_enabled=config.pipeline.restructure is not None,
    )
    sel = _select_gated_route(action, config)

    async def _cancel_candidates(except_id: str | None = None) -> None:
        for rid, st in candidates.items():
            if rid != except_id:
                await st.cancel_eager()

    # ---- parallel transfer-LLM detector (campaigns with transfer_intent /
    # transfer_llm): the referee→tool:transfer route is the preferred path; only
    # act on the dedicated detector when the gate did not already select transfer.
    transfer_llm_decision = await _resolve_transfer_llm(transfer_llm_task)
    if (
        transfer_llm_decision is not None
        and transfer_llm_decision.triggered
        and not (sel.kind == "tool" and sel.tool_type == "transfer")
    ):
        await _cancel_candidates()
        session.pipeline_trace_records.append(
            _gated_trace(
                main_stream, ts_start, user_text, None, referee_results, action,
                sel, persona_candidates, reply_suppressed=True,
            )
        )
        await _perform_handoff(
            session, sm, telephony, providers.tts,
            trigger_type=transfer_llm_decision.trigger_type or "transfer",
            trigger_detail=transfer_llm_decision.trigger_detail,
            phrase=transfer_llm_decision.phrase, voice_id=voice_id,
        )
        return Directive.RETURN

    # ---- tool routes: lazy, suppress the buffered reply ----
    if sel.kind == "tool":
        await _cancel_candidates()
        if sel.tool_type == "hangup":
            session.hangup_cause = HangupCause.REFEREE_HANGUP.value
            phrase = (sel.tool_config or {}).get("closing_phrase") or ""
            if phrase:
                await _play_tts(session, telephony, providers.tts, phrase, voice_id=voice_id)
            session.pipeline_trace_records.append(
                _gated_trace(
                    main_stream, ts_start, user_text, None, referee_results, action,
                    sel, persona_candidates, reply_suppressed=True,
                )
            )
            sm.transition_to(CallStatus.END, reason="referee_hangup", force=True)
            session.append_event("hangup", reason="referee_hangup", initiated_by="ai")
            return Directive.RETURN
        # transfer
        session.pipeline_trace_records.append(
            _gated_trace(
                main_stream, ts_start, user_text, None, referee_results, action,
                sel, persona_candidates, reply_suppressed=True,
            )
        )
        await _perform_handoff(
            session, sm, telephony, providers.tts,
            trigger_type="referee", trigger_detail="referee_decision",
            phrase="", voice_id=voice_id,
        )
        return Directive.RETURN

    # ---- restructure route: re-voice InterruptText (no new reply) ----
    if sel.route_id == "restructure":
        await _cancel_candidates()
        capped = (
            session.consecutive_restructure_count >= config.pipeline.max_continuous_restructure
        )
        restructure_text = None if capped else _assemble_interrupt_text(session, sel.source)
        session.pipeline_trace_records.append(
            _gated_trace(
                main_stream, ts_start, user_text, None, referee_results, action, sel,
                persona_candidates, restructure_text=restructure_text,
            )
        )
        if capped:
            if config.pipeline.default_replies:
                await _play_tts(
                    session, telephony, providers.tts,
                    random.choice(config.pipeline.default_replies), voice_id=voice_id,
                )
            session.consecutive_restructure_count = 0
            return Directive.CONTINUE
        if restructure_text is not None:
            played_rs = await _run_restructure(
                session, telephony, providers, config, restructure_text,
                pipeline_timeout_ms=pipeline_timeout_ms,
            )
            session.consecutive_restructure_count += 1
            if not played_rs:
                session.consecutive_interruption_count += 1
                session.interruption_signaled = False
            return Directive.CONTINUE
        return Directive.CONTINUE  # degraded (no InterruptText)

    # ---- dialogue routes: release the winning reply, cancel the losers ----
    if sel.route_id in candidates:
        winner = candidates[sel.route_id]
        winner_spec = persona_specs.get(sel.route_id)
        await _cancel_candidates(except_id=sel.route_id)
    else:
        # closing / recovery — not speculatively buffered; generate fresh now.
        await _cancel_candidates()
        winner_is_wrap_up = sel.route_id == "closing"
        winner = PipelineStream(
            session, user_text, config.pipeline, providers.llm, providers.llm,
            is_wrap_up=winner_is_wrap_up, pipeline_timeout_ms=pipeline_timeout_ms,
        )
        winner.turn_id = main_stream.turn_id
        winner_spec = None

    role_config_id = (
        winner_spec.role_config_id if winner_spec is not None
        else config.pipeline.main.role_config_id
    )

    sm.transition_to(CallStatus.IN_CALL, reason="main_stream_started")
    first_audio_ms, played = await _play_streaming(
        session, telephony, providers.tts, winner,
        voice_id=voice_id, filler=filler,
        filler_delay_s=config.filler_delay_ms / 1000.0,
        processing_start=processing_start,
    )
    session.total_tokens_in += winner.result.tokens_in
    session.total_tokens_out += winner.result.tokens_out

    goal_achieved = sel.route_id == "closing"
    goal_type = sel.goal_type if goal_achieved else ""

    if not played:
        # barge-in mid winning reply: referees already consumed; remainder was
        # captured into interrupt_remaining by _play_streaming for next-turn
        # restructure. Record + flag the interruption.
        session.pipeline_trace_records.append(
            _gated_trace(
                winner, ts_start, user_text, first_audio_ms, referee_results, action,
                sel, persona_candidates, interrupted=True,
            )
        )
        session.append_event(
            "ai_reply", text=winner.result.reply_text, turn_id=winner.turn_id,
            selected_role_config_id=role_config_id or None,
            goal_achieved=False, goal_type="", extracted={},
            is_wrap_up=False, interrupted=True,
        )
        session.consecutive_interruption_count += 1
        session.interruption_signaled = False
        return Directive.CONTINUE

    session.consecutive_interruption_count = 0
    session.consecutive_restructure_count = 0
    session.pipeline_trace_records.append(
        _gated_trace(
            winner, ts_start, user_text, first_audio_ms, referee_results, action,
            sel, persona_candidates,
        )
    )
    session.append_event(
        "ai_reply", text=winner.result.reply_text, turn_id=winner.turn_id,
        selected_role_config_id=role_config_id or None,
        goal_achieved=goal_achieved, goal_type=goal_type or "", extracted={},
        is_wrap_up=goal_achieved, interrupted=False,
    )

    # ---- project then_state (StateMachine is the sole writer) ----
    if sel.then_state == "WRAPPING_UP":  # closing → enter wrap-up
        sm.transition_to(CallStatus.IN_CALL, reason=goal_type or "goal_achieved")
        session.in_wrap_up = True
        session.wrap_up_started_at_monotonic = time.monotonic()
        session.wrap_up_started_at_wallclock = now_utc()
        session.append_event(
            "wrap_up_started",
            rounds_remaining=config.wrap_up.max_rounds,
            seconds_remaining=config.wrap_up.max_seconds,
        )
        session.append_event("goal_achieved", goal_type=goal_type, extracted={})
    # recovery (ACTIVATING) / main (LISTENING) both project to IN_CALL — no-op.
    return Directive.CONTINUE


async def _gated_wrap_up_turn(
    session: CallSession,
    sm: StateMachine,
    *,
    config: RuntimeConfig,
    telephony: TelephonyClient,
    providers: Providers,
    user_text: str,
    ts_start: object,
    processing_start: float,
    pipeline_timeout_ms: int,
) -> Directive:
    """A wrap-up turn under gate-first: simplified pipeline (no referees/gate),
    play the closing-style reply, then evaluate wrap-up exhaustion."""
    voice_id = config.voice_id
    stream = run_pipeline_stream(
        session, user_text, config.pipeline, providers.llm, providers.llm,
        is_wrap_up=True, pipeline_timeout_ms=pipeline_timeout_ms,
    )
    sm.transition_to(CallStatus.IN_CALL, reason="main_stream_started")
    first_audio_ms, played = await _play_streaming(
        session, telephony, providers.tts, stream,
        voice_id=voice_id, filler=None,
        filler_delay_s=config.filler_delay_ms / 1000.0,
        processing_start=processing_start,
    )
    session.total_tokens_in += stream.result.tokens_in
    session.total_tokens_out += stream.result.tokens_out
    session.pipeline_trace_records.append(
        {**_base_trace(stream, ts_start, user_text, first_audio_ms),
         "referee_results": [], "matched_rule": None,
         "restructure_active": False, "restructure_trigger": None,
         "restructure_source_text": None,
         "selected_route_id": "closing", "selected_route_kind": "dialogue",
         "persona_candidates": None}
    )
    session.append_event(
        "ai_reply", text=stream.result.reply_text, turn_id=stream.turn_id,
        selected_role_config_id=config.pipeline.main.role_config_id or None,
        goal_achieved=False, goal_type="", extracted={},
        is_wrap_up=True, interrupted=not played,
    )
    if not played:
        session.consecutive_interruption_count += 1
        session.interruption_signaled = False
        return Directive.CONTINUE

    session.wrap_up_round_count += 1
    wrap_decision = evaluate_wrap_up(
        rounds_so_far=session.wrap_up_round_count,
        started_at_monotonic=session.wrap_up_started_at_monotonic,
        config=config.wrap_up,
    )
    if not wrap_decision.proceed:
        await _play_tts(
            session, telephony, providers.tts, wrap_decision.closing_phrase, voice_id=voice_id
        )
        session.append_event("wrap_up_completed", reason=wrap_decision.reason)
        sm.transition_to(CallStatus.END, reason="wrap_up_completed", force=True)
        session.hangup_cause = HangupCause.WRAP_UP_COMPLETED.value
        session.append_event("hangup", reason="wrap_up_completed", initiated_by="ai")
        return Directive.RETURN
    return Directive.CONTINUE


def _gated_trace(
    stream: PipelineStream,
    ts_start: object,
    user_text: str,
    first_audio_ms: int | None,
    referee_results: list[RefereeResult],
    action: DeciderAction,
    sel: _GatedSelection,
    persona_candidates: list[str],
    *,
    restructure_text: str | None = None,
    reply_suppressed: bool = False,
    interrupted: bool = False,
) -> dict[str, Any]:
    """Build a gate-first pipeline_trace record: legacy fields + the gating
    selection columns (selected_route_id / selected_route_kind /
    persona_candidates)."""
    _ = (reply_suppressed, interrupted)  # reflected via reply_text / event, kept for clarity
    return {
        **_base_trace(stream, ts_start, user_text, first_audio_ms),
        "referee_results": [r.as_trace() for r in referee_results],
        "matched_rule": action.matched_rule,
        "restructure_active": restructure_text is not None,
        "restructure_trigger": (
            sel.restructure_trigger if restructure_text is not None else None
        ),
        "restructure_source_text": restructure_text,
        "selected_route_id": sel.route_id,
        "selected_route_kind": sel.kind,
        "persona_candidates": persona_candidates if len(persona_candidates) > 1 else None,
    }


def _assemble_interrupt_text(session: CallSession, source: str | None) -> str | None:
    """Build the restructure InterruptText for the matched source (D5).

    ``interrupt_remaining`` consumes (and clears) the captured barge-in
    remainder, degrading to ``last_reply`` when empty. ``last_reply`` (also the
    low-confidence trigger) takes the last assistant utterance in
    dialog_history. Returns ``None`` when nothing is available (→ restructure
    degrades to continue).
    """
    if source == "interrupt_remaining":
        remaining = session.interrupt_remaining_text
        session.interrupt_remaining_text = None  # take-and-clear
        if remaining and remaining.strip():
            return remaining
        source = "last_reply"  # degrade
    # last_reply (and low_confidence): last assistant utterance.
    for turn in reversed(session.dialog_history):
        if turn.role == "assistant" and turn.text.strip():
            return turn.text
    return None


def _base_trace(
    stream: PipelineStream,
    ts_start: object,
    user_text: str,
    first_audio_ms: int | None,
) -> dict[str, object]:
    """The main-LLM portion of a pipeline_trace record (referee/restructure
    fields are merged in by the caller)."""
    return {
        "turn_id": stream.turn_id,
        "ts_start": ts_start,
        "ts_end": now_utc(),
        "user_input": user_text,
        "main_reply_text": stream.result.reply_text,
        "main_duration_ms": stream.result.duration_ms,
        "main_tokens_in": stream.result.tokens_in,
        "main_tokens_out": stream.result.tokens_out,
        "main_fallback_used": stream.result.fallback_used,
        "referee_results": [],
        "matched_rule": None,
        "restructure_active": False,
        "restructure_trigger": None,
        "restructure_source_text": None,
        "first_audio_ms": first_audio_ms,
        "error": stream.result.error,
    }


async def _run_restructure(
    session: CallSession,
    telephony: TelephonyClient,
    providers: Providers,
    config: RuntimeConfig,
    interrupt_text: str,
    *,
    pipeline_timeout_ms: int,
) -> bool:
    """Run + play the restructure stream and record its ai_reply.

    Re-voices ``interrupt_text`` via the restructure slot (no referees). Returns
    whether it fully played (False on barge-in). The caller drives state.
    """
    rs = run_restructure_stream(
        session,
        interrupt_text,
        config.pipeline,
        providers.llm,
        pipeline_timeout_ms=pipeline_timeout_ms,
    )
    _first_audio, played = await _play_streaming(
        session,
        telephony,
        providers.tts,
        rs,
        voice_id=config.voice_id,
        filler=None,
        filler_delay_s=config.filler_delay_ms / 1000.0,
        processing_start=time.monotonic(),
    )
    restructure_rc_id = (
        config.pipeline.restructure.role_config_id if config.pipeline.restructure else 0
    )
    session.append_event(
        "ai_reply",
        text=rs.result.reply_text,
        turn_id=rs.turn_id,
        selected_role_config_id=restructure_rc_id or None,
        goal_achieved=False,
        goal_type="",
        extracted={},
        is_wrap_up=False,
        interrupted=not played,
    )
    return played


async def _play_chunks(
    session: CallSession,
    telephony: TelephonyClient,
    chunk_iter: AsyncIterator[bytes],
    *,
    interruptible: bool = True,
) -> bool:
    """Push a PCM chunk stream out via the telephony client.

    Returns ``True`` on full playback, ``False`` if cancelled mid-stream by
    the partial monitor (real-time interruption). The audio_out implementation
    on every TelephonyClient MUST honour ``CancelledError`` and stop pushing
    chunks (contract test enforces this).

    When ``interruptible=False`` the partial monitor is bypassed for this
    playback (used for protection cues like "您请说" that MUST complete).

    Shared by single-shot ``_play_tts`` (greeting / cues / silence phrases)
    and the producer/consumer ``_play_streaming`` pipeline
    (pipeline-latency-tail § B).
    """
    play_task = asyncio.create_task(
        telephony.audio_out(session.call_record_id, chunk_iter), name="audio_out"
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


async def _play_tts(
    session: CallSession,
    telephony: TelephonyClient,
    tts: TTSProvider,
    text: str,
    *,
    voice_id: str,
    interruptible: bool = True,
) -> bool:
    """Synthesize ``text`` and stream the PCM out (single-shot path)."""

    async def chunks() -> AsyncIterator[bytes]:
        async for chunk in tts.synthesize_stream(text, voice_id):
            yield chunk

    return await _play_chunks(
        session, telephony, chunks(), interruptible=interruptible
    )


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
        if verdict.verdict != "triggered":
            continue

        # The VAD-corroboration gate that used to sit here (require
        # vad_voice_active_ms ≥ min_duration before trusting the partial) was
        # removed 2026-06-04: its rationale was the retracted DingRTC
        # self-loopback theory, and with the persistent ASR connection the
        # ASR is reliably listening during SPEAKING, so a `triggered` partial
        # (evaluate_partial already gates on text length ≥2 + duration +
        # whitelist) is the barge-in signal. (call 143: the partial monitor
        # got the user's "1234" during SPEAKING but this gate wrongly blocked
        # the cancel.)

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
    # RMS threshold for "voice present" on the inbound mono PCM.
    # NOTE (2026-06-04): the value 1200 was originally justified as riding
    # "above the DingRTC self-loopback baseline ~600-800". That rationale is
    # RETRACTED — call 138's inbound RMS does not show a sustained loopback
    # floor, and the self-loopback causal theory is unsupported. 1200 is kept
    # AS-IS for now (changing it without fresh measurement would repeat the
    # guess-driven mistake); it is pending an evidence-based re-evaluation
    # against a controlled call (user silent during greeting → measure the
    # real inbound floor; user barge-in → measure real user-speech RMS). See
    # the interruption-detection re-eval openspec change.
    voice_rms_threshold = 1200
    # Hangover window: Chinese / Mandarin speech has natural ~50-100 ms
    # micro-pauses between syllables / words / breath. If we reset on
    # every silent frame, voice_active_ms never accumulates past the
    # threshold during natural speech. Allow up to 300 ms of contiguous
    # silence inside a voice burst before resetting the active counter —
    # well under the duration_ms threshold so an actually-silent gap of
    # 400+ ms still resets cleanly.
    silence_hangover_ms = 300
    voice_active_ms = 0
    silence_run_ms = 0
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
                silence_run_ms = 0
            else:
                # Tolerate short silence within an active burst — only
                # reset the burst counter once contiguous silence exceeds
                # the hangover window.
                if voice_active_ms > 0:
                    silence_run_ms += frame_ms
                    if silence_run_ms >= silence_hangover_ms:
                        voice_active_ms = 0
                        silence_run_ms = 0
                # Mirror onto the session so partial_monitor can corroborate.
                session.vad_voice_active_ms = voice_active_ms
                continue
            # Mirror onto the session so partial_monitor can corroborate.
            session.vad_voice_active_ms = voice_active_ms

            if session.current_speaking_task is None or session.interruption_signaled:
                # Not in SPEAKING / FILLER, or another path already fired —
                # keep accumulating voice_active_ms so the surfaced count
                # in the event reflects the user's actual span, but skip
                # the cancel side-effect.
                continue

            if voice_active_ms < interruption_cfg.min_duration_ms:
                continue

            # PROTOTYPE 2026-06-03: VAD-source cancel DISABLED.
            # VAD energy 信号容易被 ambient noise (椅子/风扇/呼吸/键盘)
            # 误触发 — 2026-06-03 真 mic smoke #128 实证 rms=1705
            # voice_active_ms=200 (接近 mac mic baseline) 误打断 AI
            # 长回应. 改用 ASR partial text length gate (≥2 字)
            # 作为唯一 barge-in 信号 (在 evaluate_partial 里),
            # 噪音过不了 vendor model 不会识别成中文文字这道闸.
            # VAD 仍 mirror voice_active_ms 给 partial_monitor 做 corroboration
            # (550902d 加的防自环误触机制保留). 验证通过后 promote 到
            # InterruptionConfig + openspec change interruption-detection-text-length-gate.
            logger.info(
                "vad_monitor_cancel_disabled_prototype "
                "voice_active_ms=%s rms=%s "
                "(barge-in 改靠 partial text length, 见 evaluate_partial)",
                voice_active_ms, rms,
            )
            continue
            # 下面原 cancel 路径暂时不走 (PROTOTYPE)
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
            session.vad_voice_active_ms = 0
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
