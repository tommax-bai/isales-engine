"""Persist a finished CallSession to ``call_record`` + ``pipeline_trace``.

Spec: transcript § 三段式存储模型; ai-pipeline (impl-engine spec delta) § per-turn
pipeline_trace persistence.

Called from the ``CallSession.run()`` END finally block (PR #11). PR #2 ships
the function shape; the wiring follows once the run loop exists.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from isales_common.enums import CallStatus, TransferStatus
from isales_common.models import CallRecord, PipelineTrace
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from isales_engine.call_session import CallSession

logger = logging.getLogger(__name__)


async def persist_call_record(
    sessionmaker: async_sessionmaker[AsyncSession],
    session: CallSession,
    *,
    started_at: datetime,
    ended_at: datetime,
    max_retries: int = 3,
) -> bool:
    """Persist call_record + pipeline_trace; return True on success.

    Retries up to ``max_retries`` with exponential backoff on transient DB
    errors. Failures log ERROR but never raise — callers (run finally block)
    must continue to LPUSH CallEnded / DECR concurrency regardless, per
    impl-engine design § 25 (error boundaries).
    """

    backoff = 0.2
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            async with sessionmaker() as db:
                await _write_in_transaction(
                    db, session, started_at=started_at, ended_at=ended_at
                )
            return True
        except Exception as exc:  # noqa: BLE001 — recorder swallows by contract
            last_exc = exc
            logger.exception(
                "transcript_persist_failed attempt=%s call_record_id=%s",
                attempt,
                session.call_record_id,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    logger.error(
        "transcript_persist_giving_up call_record_id=%s last_exc=%s",
        session.call_record_id,
        last_exc,
    )
    return False


async def _write_in_transaction(
    db: AsyncSession,
    session: CallSession,
    *,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    record = await db.get(CallRecord, session.call_record_id)
    if record is None:
        # Created by the dial consumer in PR #3; if missing here we can still
        # insert a fresh row so the call is at least audit-traceable.
        record = CallRecord(
            id=session.call_record_id,
            lead_id=session.lead_id,
            campaign_id=session.campaign_id,
            caller_id=session.caller_id,
            prompt_versions=session.prompt_versions_snapshot,
        )
        db.add(record)

    record.status = CallStatus.END
    record.started_at = started_at
    record.ended_at = ended_at
    record.duration = max(0, int((ended_at - started_at).total_seconds()))
    record.transcript = list(session.full_transcript)
    record.transfer_status = (
        TransferStatus(session.transfer_status)
        if session.transfer_status in TransferStatus.__members__.values()
        else TransferStatus.NONE
    )
    record.transfer_reason = session.transfer_reason
    if session.wrap_up_started_at_wallclock is not None:
        record.wrap_up_started_at = session.wrap_up_started_at_wallclock
    record.prompt_versions = session.prompt_versions_snapshot

    for trace_dict in session.pipeline_trace_records:
        trace = PipelineTrace(
            call_record_id=session.call_record_id,
            turn_id=int(trace_dict["turn_id"]),
            ts_start=trace_dict.get("ts_start") or started_at,
            ts_end=trace_dict.get("ts_end"),
            user_input=trace_dict.get("user_input"),
            main_reply_text=trace_dict.get("main_reply_text"),
            main_duration_ms=trace_dict.get("main_duration_ms"),
            main_tokens_in=trace_dict.get("main_tokens_in"),
            main_tokens_out=trace_dict.get("main_tokens_out"),
            main_fallback_used=bool(trace_dict.get("main_fallback_used") or False),
            referee_decision=trace_dict.get("referee_decision"),
            referee_goal_type=trace_dict.get("referee_goal_type"),
            referee_confidence=trace_dict.get("referee_confidence"),
            referee_duration_ms=trace_dict.get("referee_duration_ms"),
            first_audio_ms=trace_dict.get("first_audio_ms"),
            error=trace_dict.get("error"),
        )
        db.add(trace)

    await db.commit()


def now_utc() -> datetime:
    return datetime.now(UTC)
