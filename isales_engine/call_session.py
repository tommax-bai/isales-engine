"""Per-call session: state + dialog_history + full_transcript + counters.

Spec: call-state-machine § Implementation Notes ("each call_session 持有自己的
状态实例"); transcript § dialog_history 与 full_transcript 双集合;
silence-activation § Implementation Notes; ai-pipeline § pipeline_trace.

The :class:`CallSession` is the single shared mutable surface for one call —
state machine, orchestrator, filler / silence / interruption / transfer /
wrap-up modules all read and write it. Every ``CallSession`` is owned by the
``SessionManager`` and torn down via ``run()`` finally blocks (PR #11).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isales_common.enums import CallStatus

if TYPE_CHECKING:
    from isales_engine.state_machine import StateMachine


# Per transcript spec § Requirement: 事件类型枚举 + impl-engine spec delta.
TRANSCRIPT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "greeting",
        "user_speech",
        "ai_reply",
        "interruption",
        "filler",
        "default_reply_used",
        "silence_activation",
        "transfer_initiated",
        "transfer_marked",
        "goal_achieved",
        "wrap_up_started",
        "wrap_up_completed",
        "hangup",
        # Engine-internal events introduced by impl-engine spec deltas.
        "state_changed",
        # state_error: historical, written by transition_to pre-soften-guard.
        # state_warning: current, written when a transition falls outside
        # LEGAL_TRANSITIONS (advisory, non-blocking). See call-state-machine
        # § "非法 transition 改 advisory 警告".
        "state_error",
        "state_warning",
    }
)

# Only "real conversation" events go into dialog_history (transcript spec §
# Scenario "dialog_history 包含的事件"). Everything else stays in
# full_transcript only.
DIALOG_EVENT_TYPES: frozenset[str] = frozenset({"greeting", "user_speech", "ai_reply"})


@dataclass
class DialogTurn:
    role: str  # "assistant" for greeting / ai_reply, "user" for user_speech
    text: str
    ts_ms: int


@dataclass
class CallSession:
    """All state for one in-flight call. Owned by ``SessionManager``."""

    call_record_id: int
    campaign_id: int
    lead_id: int
    caller_id: str

    # Snapshot pinned at dispatch time (DialRequest.prompt_versions).
    prompt_versions_snapshot: dict[str, Any]

    # Scheduler-selected device for this call (DialRequest.device_id).
    # Threaded through to RtcTelephonyClient.set_device_for_session so
    # DialCommand.device_id reaches the edge.
    device_id: int = 0

    # post-call extractor (pipeline-stream-and-referee): the extractor slot ids
    # stashed by run_session so finalize_session can LPUSH the isales:extract
    # task. 0 → no extractor configured → no extract LPUSH.
    extractor_role_config_id: int = 0
    extractor_prompt_version_id: int = 0

    # Mutable state.
    state: CallStatus = CallStatus.INIT
    previous_state: CallStatus | None = None
    state_history: list[tuple[CallStatus, CallStatus, str | None]] = field(default_factory=list)

    dialog_history: list[DialogTurn] = field(default_factory=list)
    full_transcript: list[dict[str, Any]] = field(default_factory=list)
    pipeline_trace_records: list[dict[str, Any]] = field(default_factory=list)

    # Filler module bookkeeping.
    used_filler_phrase_ids_per_set: dict[int, set[int]] = field(default_factory=dict)
    current_filler_set_index: int = 0

    # Silence / no-progress / interruption counters.
    silence_activation_count: int = 0
    silence_started_at: float | None = None
    consecutive_interruption_count: int = 0
    last_user_speech_end_at: float | None = None
    last_tts_end_at: float | None = None

    # Wrap-up counters.
    wrap_up_round_count: int = 0
    wrap_up_started_at_monotonic: float | None = None

    # Pipeline turn id (incremented per PROCESSING entry).
    current_turn_id: int = 0

    # Hangup metadata, populated when entering END.
    hangup_cause: str | None = None
    transfer_status: str = "none"
    transfer_reason: str | None = None
    wrap_up_started_at_wallclock: Any = None  # datetime, set when WRAPPING_UP begins

    # Active asyncio tasks owned by the session (filler / pipeline / TTS).
    tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    # The currently-running SPEAKING / FILLER playback task. The partial
    # monitor cancels this on real-time interruption (impl-engine-providers
    # PR #6).
    current_speaking_task: asyncio.Task[Any] | None = None
    # Set by the partial monitor right before cancelling current_speaking_task;
    # the main loop reads + clears it to drive INTERRUPTED→PROCESSING.
    interruption_signaled: bool = False
    # Per-utterance speech-start timestamp (relative ms). Set when partial
    # monitor sees first non-whitelist partial; cleared on speech_end /
    # interruption.
    current_user_speech_started_ms: int | None = None
    # Updated by _vad_monitor on every inbound audio frame — the current
    # consecutive voice-active duration (with hangover tolerance). Read by
    # _partial_monitor as a corroboration signal before cancelling.
    # (The original "guards against DingRTC self-loopback ASR mis-transcription"
    # rationale is retracted 2026-06-04 — unsupported by evidence; this gate
    # is pending re-evaluation. See run_loop.py voice_rms_threshold note.)
    vad_voice_active_ms: int = 0

    # Token budget bookkeeping (impl-engine-providers PR #7).
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    # Configured per call by ``run_session``. Used by the END finally block
    # to emit the WARN log on overage; default of 0 disables the check (we
    # use 50k as the live default in run_session itself).
    _token_budget_per_call: int = 0

    # Wallclock + monotonic anchors. ``ts`` in transcript events is relative
    # to ``call_started_at_monotonic`` in milliseconds (transcript spec §
    # Requirement: 通用事件结构).
    call_started_at_monotonic: float = field(default_factory=time.monotonic)

    # Reference to the state machine driving this session. Wired by main loop
    # (``CallSession.run``) in PR #11; PR #2 leaves it ``None`` until then.
    state_machine: StateMachine | None = None

    # ---- Event recording -------------------------------------------------

    def append_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Append one event to ``full_transcript``; mirror to dialog_history when applicable.

        Auto-fills ``type`` and ``ts`` (relative ms since call_started). Raises
        ``ValueError`` if ``event_type`` is not in the transcript spec enum.
        """

        if event_type not in TRANSCRIPT_EVENT_TYPES:
            raise ValueError(f"unknown transcript event type: {event_type!r}")

        ts_ms = int((time.monotonic() - self.call_started_at_monotonic) * 1000)
        event: dict[str, Any] = {"type": event_type, "ts": ts_ms, **fields}
        self.full_transcript.append(event)

        if event_type in DIALOG_EVENT_TYPES:
            text = str(fields.get("text", ""))
            role = "user" if event_type == "user_speech" else "assistant"
            self.dialog_history.append(DialogTurn(role=role, text=text, ts_ms=ts_ms))

        return event
