"""Call state machine.

Spec: call-state-machine § Requirement: 状态集合 / 关键状态转换;
      impl-engine spec delta § Scenario "非法转移抛异常并写 state_error".

The single allowed mutation entry point is :meth:`StateMachine.transition_to`.
Any caller (orchestrator / filler / silence / transfer / wrap_up) goes through
this method so transcript events stay aligned with state changes.

Illegal transitions raise :class:`IllegalTransition` and append a
``state_error`` event to ``full_transcript``. Callers decide whether to ignore
(stay in current state) or force a terminal transition via ``force=True``
(used for ``END(reason="engine_internal_error")``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isales_common.enums import CallStatus

if TYPE_CHECKING:
    from isales_engine.call_session import CallSession


CallState = CallStatus


# Each entry maps a state to the set of states it can legally transition to.
# Sourced directly from call-state-machine spec § Requirement: 关键状态转换 +
# § Requirement: TRANSFERRING 状态的边界 + § Requirement: WRAPPING_UP 期间走简化管线.
LEGAL_TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.INIT: {
        CallState.GREETING,
        CallState.END,  # dial fail / no_answer / etc.
    },
    CallState.GREETING: {
        CallState.LISTENING,
        CallState.END,  # remote_hangup mid-greeting
    },
    CallState.LISTENING: {
        CallState.PROCESSING,
        CallState.ACTIVATING,
        CallState.TRANSFERRING,
        CallState.WRAPPING_UP,  # rare: explicit goal_achieved before any user speech
        CallState.END,
    },
    CallState.SPEAKING: {
        CallState.LISTENING,  # TTS finished cleanly
        CallState.INTERRUPTED,
        CallState.WRAPPING_UP,  # post goal_achieved reply finished playing
        CallState.TRANSFERRING,
        CallState.END,
    },
    CallState.INTERRUPTED: {
        CallState.PROCESSING,
        CallState.END,
    },
    CallState.FILLER: {
        # FILLER runs concurrently with PROCESSING and naturally returns to
        # SPEAKING when the pipeline finishes, or to PROCESSING when the user
        # interrupts the filler audio.
        CallState.SPEAKING,
        CallState.PROCESSING,
        CallState.END,
    },
    CallState.PROCESSING: {
        CallState.SPEAKING,
        CallState.LISTENING,  # default reply skipping speak phase is legal too
        CallState.TRANSFERRING,
        CallState.WRAPPING_UP,
        CallState.END,
    },
    CallState.WRAPPING_UP: {
        # Wrap-up runs its own simplified pipeline: PROCESSING / SPEAKING are
        # entered in the wrap-up loop; counter exhaustion ends the call.
        CallState.PROCESSING,
        CallState.SPEAKING,
        CallState.LISTENING,
        CallState.TRANSFERRING,
        CallState.END,
    },
    CallState.ACTIVATING: {
        CallState.LISTENING,
        CallState.END,
    },
    CallState.TRANSFERRING: {
        CallState.END,
    },
    CallState.END: set(),
}


class IllegalTransition(RuntimeError):
    """Raised when a non-``force`` ``transition_to`` violates ``LEGAL_TRANSITIONS``."""

    def __init__(self, from_state: CallState, to_state: CallState, reason: str | None = None):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f"illegal transition {from_state.value} -> {to_state.value}"
            + (f" (reason={reason})" if reason else "")
        )


class StateMachine:
    """Drives ``CallSession.state`` through the legal transition graph."""

    def __init__(self, session: CallSession) -> None:
        self._session = session

    def transition_to(
        self,
        new_state: CallState,
        *,
        reason: str | None = None,
        meta: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        """Move ``session.state`` to ``new_state``.

        On illegal transitions, append ``state_error`` to ``full_transcript``
        and raise :class:`IllegalTransition`. ``force=True`` skips the legality
        check (used for unrecoverable shutdowns into ``END``).
        """

        current = self._session.state
        if not force and new_state not in LEGAL_TRANSITIONS.get(current, set()):
            self._session.append_event(
                "state_error",
                attempted=reason or "transition",
                from_state=current.value,
                to_state=new_state.value,
            )
            raise IllegalTransition(current, new_state, reason=reason)

        self._session.previous_state = current
        self._session.state = new_state
        self._session.state_history.append((current, new_state, reason))

        # transcript hangup event is the responsibility of the END caller
        # (it carries reason + initiated_by); state machine only logs the
        # transition for non-END states to avoid duplication.
        if new_state is not CallState.END and meta is not None:
            self._session.append_event(
                "state_changed",
                from_state=current.value,
                to_state=new_state.value,
                reason=reason,
                **meta,
            )
