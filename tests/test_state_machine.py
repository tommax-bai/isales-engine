"""Tests for state_machine + CallSession + SessionManager."""

from __future__ import annotations

import logging

import pytest
from isales_common.enums import CallStatus

from isales_engine.call_session import CallSession
from isales_engine.session_manager import SessionManager
from isales_engine.state_machine import IllegalTransition, StateMachine


def make_session(call_record_id: int = 1) -> CallSession:
    return CallSession(
        call_record_id=call_record_id,
        campaign_id=42,
        lead_id=7,
        caller_id="+8613800000000",
        prompt_versions_snapshot={"main_llm": None, "referee_llm": None, "extractor_llm": None},
    )


# ---- StateMachine ----------------------------------------------------------


def test_init_to_greeting_legal() -> None:
    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.GREETING, reason="connected")
    assert s.state == CallStatus.GREETING
    assert s.previous_state == CallStatus.INIT


def test_full_normal_path() -> None:
    s = make_session()
    sm = StateMachine(s)
    for state, reason in [
        (CallStatus.GREETING, "connected"),
        (CallStatus.LISTENING, "greeting_done"),
        (CallStatus.PROCESSING, "speech_end"),
        (CallStatus.SPEAKING, "pipeline_done"),
        (CallStatus.LISTENING, "tts_done"),
        (CallStatus.PROCESSING, "speech_end"),
        (CallStatus.SPEAKING, "pipeline_done"),
        (CallStatus.WRAPPING_UP, "goal_achieved"),
        (CallStatus.END, "wrap_up_completed"),
    ]:
        sm.transition_to(state, reason=reason)
    assert s.state == CallStatus.END


def test_unusual_transition_writes_warning_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Spec call-state-machine § 非法 transition 改 advisory 警告 / Scenario 非常规 transition 不抛异常."""

    s = make_session()
    sm = StateMachine(s)

    with caplog.at_level(logging.WARNING, logger="isales_engine.state_machine"):
        # INIT -> SPEAKING is not in LEGAL_TRANSITIONS — should NOT raise.
        sm.transition_to(CallStatus.SPEAKING, reason="bug")

    # State actually advanced.
    assert s.state == CallStatus.SPEAKING
    assert s.previous_state == CallStatus.INIT
    assert s.state_history[-1] == (CallStatus.INIT, CallStatus.SPEAKING, "bug")

    # Transcript got a state_warning event (NOT state_error).
    assert any(e["type"] == "state_warning" for e in s.full_transcript)
    assert not any(e["type"] == "state_error" for e in s.full_transcript)

    # logger.warning was emitted with a greppable signature.
    assert any(
        "state_transition_unusual" in record.message
        and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_state_warning_event_payload() -> None:
    """Spec call-state-machine § 非法 transition 改 advisory 警告 — transcript payload shape."""

    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.SPEAKING, reason="bug")

    warn = next(e for e in s.full_transcript if e["type"] == "state_warning")
    assert warn["attempted"] == "bug"
    assert warn["from_state"] == "init"
    assert warn["to_state"] == "speaking"


def test_force_true_backward_compat() -> None:
    """Spec call-state-machine § force=True 参数 backward-compat.

    force=True remains accepted (no signature change) but the behavior is
    equivalent to not passing force — both paths complete the transition.
    """

    s = make_session()
    sm = StateMachine(s)

    # First reach END (legal from INIT).
    sm.transition_to(CallStatus.END, reason="dial_fail")
    assert s.state == CallStatus.END

    # END -> GREETING is unusual (END has no outgoing transitions). With
    # force=True, transition still completes via the advisory path — same as
    # if force had been omitted.
    sm.transition_to(CallStatus.GREETING, reason="forced", force=True)
    assert s.state == CallStatus.GREETING

    # The advisory event was still written even with force=True.
    assert any(
        e["type"] == "state_warning" and e["from_state"] == "end" and e["to_state"] == "greeting"
        for e in s.full_transcript
    )


def test_unusual_transition_from_terminal_end_does_not_raise() -> None:
    """Spec call-state-machine § IllegalTransition 类保留但不再抛出."""

    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.END, reason="dial_fail")
    # END has no outgoing transitions — pre-soften this raised. Now advisory.
    sm.transition_to(CallStatus.LISTENING, reason="bug")
    assert s.state == CallStatus.LISTENING


def test_unusual_transition_from_transferring_does_not_raise() -> None:
    """TRANSFERRING -> PROCESSING is not in LEGAL_TRANSITIONS but no longer raises."""

    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.GREETING, reason="connected")
    sm.transition_to(CallStatus.LISTENING, reason="greeting_done")
    sm.transition_to(CallStatus.TRANSFERRING, reason="keyword_hit")
    sm.transition_to(CallStatus.PROCESSING, reason="bug")
    assert s.state == CallStatus.PROCESSING
    sm.transition_to(CallStatus.END, reason="marked_for_handoff")
    assert s.state == CallStatus.END


def test_illegal_transition_class_still_importable() -> None:
    """Spec call-state-machine § IllegalTransition 类可被 import 但不会被抛出."""

    # Class definition retained.
    assert issubclass(IllegalTransition, RuntimeError)
    # Can still be constructed (in case any caller code still does so).
    exc = IllegalTransition(CallStatus.INIT, CallStatus.SPEAKING, reason="bug")
    assert "init -> speaking" in str(exc)
    assert exc.reason == "bug"


def test_meta_dict_emits_state_changed_event() -> None:
    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.GREETING, reason="connected", meta={"source": "modem"})
    sc = next(e for e in s.full_transcript if e["type"] == "state_changed")
    assert sc["from_state"] == "init"
    assert sc["to_state"] == "greeting"
    assert sc["source"] == "modem"


# ---- CallSession.append_event ---------------------------------------------


def test_append_event_routes_dialog_only() -> None:
    s = make_session()
    s.append_event("greeting", text="您好我是 AI", audio_duration_ms=1200)
    s.append_event("user_speech", text="你好", asr_confidence=0.95, duration_ms=500)
    s.append_event("ai_reply", text="OK", turn_id=1, goal_achieved=False)
    s.append_event("filler", text="让我看一下", filler_phrase_id=10, duration_ms=400)
    s.append_event("silence_activation", text="还在吗", activation_index=0)

    assert len(s.full_transcript) == 5
    assert len(s.dialog_history) == 3
    assert [t.role for t in s.dialog_history] == ["assistant", "user", "assistant"]
    assert s.dialog_history[1].text == "你好"


def test_append_event_unknown_type_raises() -> None:
    s = make_session()
    with pytest.raises(ValueError):
        s.append_event("not_a_real_type", text="x")


def test_append_event_ts_relative_and_monotonic() -> None:
    s = make_session()
    s.append_event("greeting", text="hi")
    s.append_event("user_speech", text="hello")
    a = s.full_transcript[0]["ts"]
    b = s.full_transcript[1]["ts"]
    assert a >= 0
    assert b >= a


# ---- SessionManager --------------------------------------------------------


def test_session_manager_register_unregister() -> None:
    sm_mgr = SessionManager()
    s1 = make_session(1)
    s2 = make_session(2)
    sm_mgr.register(s1)
    sm_mgr.register(s2)
    assert sm_mgr.active_count() == 2
    assert sm_mgr.get(1) is s1
    assert sm_mgr.unregister(1) is s1
    assert sm_mgr.get(1) is None
    assert sm_mgr.active_count() == 1


def test_session_manager_register_idempotent() -> None:
    sm_mgr = SessionManager()
    s1 = make_session(1)
    sm_mgr.register(s1)
    sm_mgr.register(s1)  # no-op + WARN, not an exception
    assert sm_mgr.active_count() == 1


async def test_session_manager_cancel_all_waits_for_finalisers() -> None:
    import asyncio

    sm_mgr = SessionManager()
    s = make_session(1)
    sm_mgr.register(s)

    finished = asyncio.Event()

    async def long_running() -> None:
        try:
            await asyncio.sleep(60)
        finally:
            # Sync-only finalizer — async work after cancellation re-raises
            # CancelledError. Real CallSession.run() uses asyncio.shield
            # around DB writes / LPUSH / DECR; here we only need to prove
            # cancel_all awaits task completion.
            finished.set()

    s.tasks["dummy"] = asyncio.create_task(long_running())
    # Let the task actually enter `try:` before we cancel it. Without this
    # yield, cancellation marks a not-yet-started task as cancelled and the
    # body / finally never runs — which would silently bypass our assertion.
    await asyncio.sleep(0)

    await sm_mgr.cancel_all(timeout_s=2.0)
    assert finished.is_set()
