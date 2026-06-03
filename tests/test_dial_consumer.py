"""Tests for dial_consumer + call_lifecycle.

Real PG + Redis required (skip otherwise via conftest fixtures).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from isales_common.models import (
    Campaign,
    Lead,
    PipelineTrace,
)
from isales_common.schemas.messages.dial import (
    DialLeadInfo,
    DialRequest,
    PromptVersionsSnapshot,
)
from sqlalchemy import select

from isales_engine.call_lifecycle import finalize_session
from isales_engine.call_session import CallSession
from isales_engine.dial_consumer import handle_dial
from isales_engine.session_manager import SessionManager
from isales_engine.settings import Settings

# Required for fixtures from conftest.py + share session loop with the
# session-scoped engine/redis fixtures.
pytestmark = [
    pytest.mark.usefixtures("clean_engine", "redis_client"),
    pytest.mark.asyncio(loop_scope="session"),
]


def _make_settings() -> Settings:
    return Settings(
        ISALES_DATABASE_URL="postgresql://x/y",
        ISALES_REDIS_URL="redis://localhost:6379/2",
    )


async def _seed_campaign_and_lead(sessionmaker_: Any) -> tuple[int, int]:
    async with sessionmaker_() as db:
        campaign = Campaign(name="t", default_replies=["hi"])
        db.add(campaign)
        await db.flush()
        lead = Lead(campaign_id=campaign.id, phone="+8613800000000")
        db.add(lead)
        await db.commit()
        await db.refresh(campaign)
        await db.refresh(lead)
        return campaign.id, lead.id


def _build_dial_request(campaign_id: int, lead_id: int) -> DialRequest:
    return DialRequest(
        lead=DialLeadInfo(
            lead_id=lead_id, campaign_id=campaign_id, phone="+8613800000000", name="t"
        ),
        history=[],
        prompt_versions=PromptVersionsSnapshot(),
        caller_id="+8613900000000",
        device_id=1,
    )


# ---- DLQ paths -------------------------------------------------------------


async def test_dial_dlq_unknown_schema_version(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()
    session_seen: list[CallSession] = []

    async def runner(session: CallSession) -> None:
        session_seen.append(session)

    raw = '{"schema_version": 99, "message_id": "x", "created_at": "2026-01-01T00:00:00Z"}'
    result = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert result is None
    assert sm.active_count() == 0
    dlq_len = await redis_client.llen(settings.engine_dlq)
    assert dlq_len == 1


async def test_dial_dlq_invalid_json(sessionmaker_: Any, redis_client: Any) -> None:
    settings = _make_settings()
    sm = SessionManager()

    async def runner(session: CallSession) -> None:
        pass

    raw = "not-json-at-all"
    result = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert result is None
    dlq_len = await redis_client.llen(settings.engine_dlq)
    assert dlq_len == 1


async def test_dial_dlq_validation_error(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()

    async def runner(session: CallSession) -> None:
        pass

    # schema_version=1 but missing required fields → ValidationError → DLQ.
    raw = (
        f'{{"schema_version": 1, "message_id": "{uuid4()}", '
        '"created_at": "2026-01-01T00:00:00Z"}'
    )
    result = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert result is None
    dlq_len = await redis_client.llen(settings.engine_dlq)
    assert dlq_len == 1


# ---- Happy path ------------------------------------------------------------


async def test_dial_happy_path_creates_session_and_persists(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()
    campaign_id, lead_id = await _seed_campaign_and_lead(sessionmaker_)
    runner_seen: list[CallSession] = []

    async def runner(session: CallSession) -> None:
        runner_seen.append(session)
        # Simulate a normal call ending cleanly.
        session.hangup_cause = "normal_clearing"
        session.full_transcript.append({"type": "greeting", "ts": 0, "text": "hi"})
        session.pipeline_trace_records.append(
            {
                "turn_id": 1,
                "ts_start": None,
                "ts_end": None,
                "user_input": "test",
                "main_reply_text": "好的，请稍等。",
                "main_duration_ms": 120,
                "main_tokens_in": 16,
                "main_tokens_out": 8,
                "main_fallback_used": False,
                "referee_decision": "continue",
                "referee_goal_type": None,
                "referee_confidence": 0.9,
                "referee_duration_ms": 80,
                "first_audio_ms": 200,
                "error": None,
            }
        )

    raw = _build_dial_request(campaign_id, lead_id).model_dump_json()
    session = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert session is not None

    # Wait for runner+finalize to complete.
    await session.tasks["main"]
    assert len(runner_seen) == 1
    assert runner_seen[0] is session

    # call_record persisted with terminal state.
    async with sessionmaker_() as db:
        from isales_common.models import CallRecord

        rec = await db.get(CallRecord, session.call_record_id)
        assert rec is not None
        assert rec.status == "end"
        assert rec.duration is not None
        assert any(e["type"] == "greeting" for e in rec.transcript)

        traces = (
            await db.execute(
                select(PipelineTrace).where(PipelineTrace.call_record_id == rec.id)
            )
        ).scalars().all()
        assert len(traces) == 1
        # Dual-LLM trace: main reply text recorded (was polish_output).
        assert traces[0].main_reply_text
        assert traces[0].first_audio_ms is not None

    # CallEnded LPUSHed.
    queue_len = await redis_client.llen(settings.engine_call_ended_queue)
    assert queue_len == 1

    # Concurrency counter snapped back to zero (no INCR happened, but DECR
    # below zero is clamped to 0).
    val = await redis_client.get(settings.engine_concurrency_key)
    assert int(val or 0) == 0

    # Session unregistered.
    assert sm.active_count() == 0


async def test_dial_runner_exception_still_finalizes(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()
    campaign_id, lead_id = await _seed_campaign_and_lead(sessionmaker_)

    async def runner(session: CallSession) -> None:
        raise RuntimeError("boom")

    raw = _build_dial_request(campaign_id, lead_id).model_dump_json()
    session = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert session is not None
    await session.tasks["main"]

    # Even on runner failure, CallEnded is dispatched + counter DECRed +
    # session unregistered.
    queue_len = await redis_client.llen(settings.engine_call_ended_queue)
    assert queue_len == 1
    assert sm.active_count() == 0


async def test_finalize_session_lpushes_with_correct_shape(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()
    campaign_id, lead_id = await _seed_campaign_and_lead(sessionmaker_)

    raw = _build_dial_request(campaign_id, lead_id).model_dump_json()

    async def runner(session: CallSession) -> None:
        session.hangup_cause = "wrap_up_completed"

    session = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert session is not None
    await session.tasks["main"]

    payload = await redis_client.lrange(settings.engine_call_ended_queue, 0, -1)
    assert len(payload) == 1
    import json

    parsed = json.loads(payload[0])
    assert parsed["call_record_id"] == session.call_record_id
    assert parsed["hangup_cause"] == "wrap_up_completed"
    assert parsed["schema_version"] == 1


async def test_concurrency_counter_decrement(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()
    campaign_id, lead_id = await _seed_campaign_and_lead(sessionmaker_)

    # Pre-INCR to mimic scheduler's dispatch-time bump.
    await redis_client.set(settings.engine_concurrency_key, 1)

    async def runner(session: CallSession) -> None:
        session.hangup_cause = "normal_clearing"

    raw = _build_dial_request(campaign_id, lead_id).model_dump_json()
    session = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert session is not None
    await session.tasks["main"]

    val = await redis_client.get(settings.engine_concurrency_key)
    assert int(val) == 0


# ---- finalize_session direct call -----------------------------------------


async def test_finalize_session_direct(
    sessionmaker_: Any, redis_client: Any
) -> None:
    """finalize_session is callable outside the dial path (used by graceful shutdown)."""

    from isales_engine.call_lifecycle import reserve_call_record
    from isales_engine.transcript_recorder import now_utc

    settings = _make_settings()
    sm = SessionManager()
    campaign_id, lead_id = await _seed_campaign_and_lead(sessionmaker_)

    request = _build_dial_request(campaign_id, lead_id)
    started_at = now_utc()
    crid = await reserve_call_record(sessionmaker_, request, started_at=started_at)
    session = CallSession(
        call_record_id=crid,
        campaign_id=campaign_id,
        lead_id=lead_id,
        caller_id="+8613900000000",
        prompt_versions_snapshot=request.prompt_versions.model_dump(mode="json"),
    )
    sm.register(session)
    session.hangup_cause = "manual_hangup"
    session.transfer_status = "marked_for_handoff"
    session.transfer_reason = "keyword"

    await finalize_session(
        session,
        sessionmaker=sessionmaker_,
        redis=redis_client,
        settings=settings,
        session_manager=sm,
        started_at=started_at,
    )

    async with sessionmaker_() as db:
        from isales_common.models import CallRecord

        rec = await db.get(CallRecord, crid)
        assert rec is not None
        assert rec.transfer_status == "marked_for_handoff"
        assert rec.transfer_reason == "keyword"

    queue_len = await redis_client.llen(settings.engine_call_ended_queue)
    assert queue_len == 1
    assert sm.active_count() == 0


# ---- Cancellation path ----------------------------------------------------


async def test_dial_runner_cancellation_still_finalizes(
    sessionmaker_: Any, redis_client: Any
) -> None:
    settings = _make_settings()
    sm = SessionManager()
    campaign_id, lead_id = await _seed_campaign_and_lead(sessionmaker_)

    async def runner(session: CallSession) -> None:
        await asyncio.sleep(60)  # long-running; will be cancelled

    raw = _build_dial_request(campaign_id, lead_id).model_dump_json()
    session = await handle_dial(
        raw,
        redis=redis_client,
        sessionmaker=sessionmaker_,
        settings=settings,
        session_manager=sm,
        runner=runner,
    )
    assert session is not None

    # Let the runner enter `await sleep(60)` first.
    await asyncio.sleep(0)
    session.tasks["main"].cancel()

    with pytest.raises(asyncio.CancelledError):
        await session.tasks["main"]

    # Finalize ran (CallEnded LPUSHed, session unregistered).
    queue_len = await redis_client.llen(settings.engine_call_ended_queue)
    assert queue_len == 1
    assert sm.active_count() == 0
