"""Frozen-contract tripwires — change-0 safety net for the flat refactor.

These lock the *immovable* external/contract surface that the EventBus /
SelectRouter / flatten refactor MUST NOT break (see
``isales/openspec/engine-flat-refactor-blueprint.md`` §6 hard-constraints):

- ``run_session()``'s call signature (every isales-engine e2e test + dial_consumer
  drive it; the refactor freezes this contract).
- ``TRANSCRIPT_EVENT_TYPES`` — the transcript event vocabulary consumed by
  isales-worker / isales-web. Dropping or renaming one breaks downstream silently.
- ``CallStatus`` — the 11-value enum projected to api/web/transcript/worker. In
  the flat model CallStatus becomes a *projected label*, but the enum stays closed.
- ``HangupCause`` — so adding ``REFEREE_HANGUP`` (change-3) is a deliberate,
  test-visible act, and the worker's enum-validated CallEnded boundary is guarded.

A failure here is a contract change, not a bug — update the test *and* the spec
in the same change, intentionally.

NOTE (follow-up within change-0): the finalize side-effect contract (DECR
concurrency snap-to-0, dual LPUSH isales:extract / isales:call_ended exactly once
under a shielded/cancelled teardown) lives at the dial_consumer/call_lifecycle
layer and needs a Redis double — tracked as a change-0 follow-up, not covered here.
"""

from __future__ import annotations

import inspect

from isales_common.enums import CallStatus, HangupCause

from isales_engine.call_session import TRANSCRIPT_EVENT_TYPES
from isales_engine.run_loop import run_session


def test_run_session_signature_frozen() -> None:
    """run_session's signature is the frozen contract every caller depends on."""
    sig = inspect.signature(run_session)
    assert list(sig.parameters) == [
        "session",
        "phone",
        "config",
        "telephony",
        "providers",
        "publisher",
        "pipeline_timeout_ms",
        "connect_timeout_s",
        "token_budget_per_call",
    ]
    # Defaults that callers (dial_consumer) rely on.
    assert sig.parameters["publisher"].default is None
    assert sig.parameters["pipeline_timeout_ms"].default == 8000
    assert sig.parameters["connect_timeout_s"].default == 30.0
    assert sig.parameters["token_budget_per_call"].default == 50_000
    assert sig.return_annotation in (None, "None")


def test_transcript_event_vocabulary_frozen() -> None:
    """The transcript event vocabulary is an external contract (worker/web)."""
    assert TRANSCRIPT_EVENT_TYPES == frozenset(
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
            "state_changed",
            "state_error",
            "state_warning",
        }
    )


def test_call_status_enum_frozen() -> None:
    """CallStatus is the collapsed 4-value coarse lifecycle (engine-tools-
    multidialogue-gating). The fine in-call phases (greeting/listening/speaking/
    interrupted/filler/processing/wrapping_up/activating) are no longer call
    states — they are engine-internal flags + transcript events. ``in_call``
    spans the whole conversation; only handoff + the bookends are distinct."""
    assert {s.value for s in CallStatus} == {
        "init",
        "in_call",
        "transferring",
        "end",
    }


def test_hangup_cause_enum_frozen() -> None:
    """HangupCause baseline — adding REFEREE_HANGUP (change-3) must be deliberate.

    When change-3 adds ``referee_hangup`` this test is updated in the SAME change
    that bumps isales-common + the worker pin (the enum-validated CallEnded
    boundary), making the cross-repo lockstep visible.
    """
    assert {c.value for c in HangupCause} == {
        # GSM-side
        "no_answer",
        "user_busy",
        "network_out_of_order",
        "temporary_failure",
        "normal_clearing",
        "call_rejected",
        # Application-side
        "user_hangup",
        "wrap_up_completed",
        "silence_max_reached",
        "marked_for_handoff",
        "no_progress_timeout",
        "manual_hangup",
        # change-3: referee gating may terminate the call (pre-reply hangup tool
        # route). Added in lockstep with isales-common 0.8.0 + the worker pin
        # (>=0.8) so the enum-validated CallEnded boundary accepts it (no DLQ).
        "referee_hangup",
    }
