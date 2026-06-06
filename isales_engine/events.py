"""Typed intra-call event vocabulary for the in-process :class:`EventBus`.

These are cheap ``@dataclass(slots=True, frozen=True)`` payloads — deliberately
SEPARATE from the pydantic ``EngineEvent`` Redis-wire union (``isales_common``),
so the hot path stays off pydantic validation and bus events may carry values
that never cross Redis. A bridge subscriber adapts the publishable subset to the
``EngineEvent`` wire contract (later phase).

change-1 scope = the SIGNAL events the legacy ``run_loop`` coordination
primitives (``asr_finals_q`` / ``asr_partials_q`` / ``hangup_event`` /
``asr_finished`` / the partial/VAD monitors / the manual-hangup sentinel) are
being migrated onto. Route / lifecycle / status events arrive in later changes.

Lane assignment (see :mod:`isales_engine.eventbus`): ``*Final`` / ``*Requested``
/ ``*Detected`` / ``Connected`` / ``*Ended`` default LOSSLESS; ``AsrPartial`` /
``VadFrame`` are registered LOSSY by the session wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

# ----- LOSSLESS signals (never drop) --------------------------------------- #


@dataclass(slots=True, frozen=True)
class AsrFinal:
    """A finalized user utterance (was ``asr_finals_q.put``). Lossless."""

    text: str


@dataclass(slots=True, frozen=True)
class AsrStreamEnded:
    """The ASR stream terminated (was ``asr_finished.set()``). Lossless."""


@dataclass(slots=True, frozen=True)
class Connected:
    """The dialed call connected. Lossless."""


@dataclass(slots=True, frozen=True)
class HangupDetected:
    """Inbound hangup from telephony (remote/local hangup, device error).

    ``cause`` is an ``isales_common.enums.HangupCause`` value string. Lossless.
    """

    cause: str


@dataclass(slots=True, frozen=True)
class InterruptRequested:
    """A barge-in was detected (was the partial/VAD monitor reaching across to
    ``session.current_speaking_task.cancel()``).

    ``source`` ∈ {``"partial"``, ``"vad"``}. ``user_text`` is the triggering
    partial; the playback owner subscribes and cancels ITSELF, capturing the
    not-yet-spoken remainder. Lossless (interruption 判定不可撤销).
    """

    source: str
    user_text: str | None = None


@dataclass(slots=True, frozen=True)
class ManualHangupRequested:
    """Operator-initiated hangup arriving over Redis control (replaces the
    ``request_manual_hangup`` ``main.cancel()`` sentinel hack). Lossless."""

    operator: str | None = None


@dataclass(slots=True, frozen=True)
class TransferRequested:
    """Operator/control-initiated transfer-to-human. Lossless."""

    agent_id: str | None = None


# ----- LOSSY signals (drop-oldest acceptable) ------------------------------ #


@dataclass(slots=True, frozen=True)
class AsrPartial:
    """A non-final (partial) ASR hypothesis (was ``asr_partials_q.put``). Lossy."""

    text: str


@dataclass(slots=True, frozen=True)
class VadFrame:
    """A VAD/RMS frame for the (currently-disabled) VAD barge-in path. Lossy."""

    rms: float
    voice_active_ms: int
