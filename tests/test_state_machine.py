"""Tests for state_machine + CallSession + SessionManager."""

from __future__ import annotations

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
        prompt_versions_snapshot={"role_llms": [], "judge_llm": None, "polish_llm": None},
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


def test_illegal_transition_raises_and_logs_state_error() -> None:
    s = make_session()
    sm = StateMachine(s)
    # INIT -> SPEAKING is not in LEGAL_TRANSITIONS.
    with pytest.raises(IllegalTransition):
        sm.transition_to(CallStatus.SPEAKING, reason="bug")

    assert s.state == CallStatus.INIT  # state unchanged
    assert any(e["type"] == "state_error" for e in s.full_transcript)
    err = next(e for e in s.full_transcript if e["type"] == "state_error")
    assert err["from_state"] == "init"
    assert err["to_state"] == "speaking"


def test_force_skips_legality_check() -> None:
    s = make_session()
    sm = StateMachine(s)
    # WRAPPING_UP -> END is legal; INIT -> END is also legal — pick a really
    # illegal one: END -> GREETING.
    sm.transition_to(CallStatus.END, reason="dial_fail")
    sm.transition_to(CallStatus.GREETING, reason="forced", force=True)
    assert s.state == CallStatus.GREETING


def test_terminal_end_has_no_outgoing_transitions() -> None:
    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.END, reason="dial_fail")
    with pytest.raises(IllegalTransition):
        sm.transition_to(CallStatus.LISTENING, reason="bug")


def test_transferring_only_to_end() -> None:
    s = make_session()
    sm = StateMachine(s)
    sm.transition_to(CallStatus.GREETING, reason="connected")
    sm.transition_to(CallStatus.LISTENING, reason="greeting_done")
    sm.transition_to(CallStatus.TRANSFERRING, reason="keyword_hit")
    with pytest.raises(IllegalTransition):
        sm.transition_to(CallStatus.PROCESSING, reason="bug")
    sm.transition_to(CallStatus.END, reason="marked_for_handoff")


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
