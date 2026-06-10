"""Call state machine — the 4-state lifecycle + the sole state writer.

Spec: call-state-machine § Requirement: 状态集合 / 关键状态转换 /
      § "StatusProjector 单写者投影状态" / § "非法 transition 改 advisory 警告".

This class IS the spec's "StatusProjector": the single, **synchronous** writer of
``session.state`` + sole emitter of ``StatusChanged``. The only mutation entry
point is :meth:`StateMachine.transition_to`; the gate-first turn projects a
route's ``then_state`` through it (routes never write state directly). Post the
4-state collapse the observable states are just {init, in_call, transferring,
end}, so ``transition_to`` is idempotent and most legacy intra-call transitions
fold to a no-op — a sync sole-writer needs no async bus subscriber.

Transition guard is **advisory**: a transition not listed in
``LEGAL_TRANSITIONS`` emits a ``state_warning`` transcript event +
``logger.warning("state_transition_unusual ...")`` log line, but **does NOT
block the transition** — ``session.state`` is updated unconditionally.
``IllegalTransition`` is retained for import compatibility but is never raised.
Rationale: see spec call-state-machine § "非法 transition 改 advisory 警告".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isales_common.enums import CallStatus

if TYPE_CHECKING:
    from isales_engine.call_session import CallSession


logger = logging.getLogger(__name__)


CallState = CallStatus


# Each entry maps a state to the set of states it can legally transition to.
# engine-tools-multidialogue-gating collapsed the 11-value FSM-as-controller set
# to 4 coarse lifecycle labels: the whole conversation is ``IN_CALL``; the fine
# phases (greeting / listening / processing / speaking / interrupted / filler /
# wrapping_up / activating) are now engine-internal flags (e.g.
# ``session.in_wrap_up``) + transcript events, NOT call states. Only a
# human-handoff (``TRANSFERRING``) and the bookends (``INIT`` / ``END``) are
# distinct. ``transition_to`` is idempotent (same-state → no-op) so the many
# legacy intra-call transitions that now all map to ``IN_CALL`` collapse to a
# single ``INIT → IN_CALL`` with no history/event churn.
LEGAL_TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.INIT: {
        CallState.IN_CALL,
        CallState.END,  # dial fail / no_answer / etc.
    },
    CallState.IN_CALL: {
        CallState.TRANSFERRING,
        CallState.END,
    },
    CallState.TRANSFERRING: {
        CallState.END,
    },
    CallState.END: set(),
}


class IllegalTransition(RuntimeError):
    """Retained for import compatibility; ``transition_to`` no longer raises it.

    Kept as a class definition so existing ``except IllegalTransition`` blocks
    (if any) and ``from isales_engine.state_machine import IllegalTransition``
    imports keep working. A followup change cleans up dead handlers.
    """

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
        force: bool = False,
    ) -> None:
        """Move ``session.state`` to ``new_state``.

        Transitions not present in ``LEGAL_TRANSITIONS[current]`` are
        **advisory-only**: a ``state_warning`` event is appended to
        ``full_transcript`` and a ``state_transition_unusual`` warning is
        logged, but the transition still completes. ``force=True`` is retained
        for backward-compat with the 8 existing END-shutdown callers but no
        longer has any behavioral effect (the guard never blocks).
        """

        current = self._session.state
        if new_state == current:
            # Idempotent: the 4-state collapse maps the many legacy intra-call
            # transitions (listening/processing/speaking/...) all to IN_CALL, so
            # most calls are now same-state no-ops. Skip them so state_history /
            # previous_state / StatusChanged don't churn on a non-event.
            return
        if new_state not in LEGAL_TRANSITIONS.get(current, set()):
            logger.warning(
                "state_transition_unusual current=%s new=%s reason=%s",
                current.value,
                new_state.value,
                reason,
            )
            self._session.append_event(
                "state_warning",
                attempted=reason or "transition",
                from_state=current.value,
                to_state=new_state.value,
            )

        self._session.previous_state = current
        self._session.state = new_state
        self._session.state_history.append((current, new_state, reason))
