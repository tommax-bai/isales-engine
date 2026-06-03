"""Pre-dial call_record creation + post-END finalization.

Spec: service-communication § engine→worker (CallEnded LPUSH); §
全局并发控制 (DECR ``isales:concurrency:active`` on call end);
message-contract § Scenario "v1 必有的消息类" (CallEnded shape).

The dial consumer (``dial_consumer.handle_dial``) reserves a fresh
``call_record`` row at dispatch time so other services can see status / ids
during the call. ``finalize_session`` runs from the per-call task's finally
block: persists transcript / pipeline_trace via ``transcript_recorder``,
LPUSHes ``CallEnded`` to the worker queue, DECRs the global concurrency
counter, and unregisters from ``SessionManager``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from isales_common.enums import CallStatus, HangupCause
from isales_common.models import CallRecord
from isales_common.redis_keys import EXTRACT_QUEUE
from isales_common.schemas.messages import CallEnded
from isales_common.schemas.messages.dial import DialRequest
from redis.asyncio import Redis
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from isales_engine.call_session import CallSession
from isales_engine.session_manager import SessionManager
from isales_engine.settings import Settings
from isales_engine.transcript_recorder import now_utc, persist_call_record

logger = logging.getLogger(__name__)


async def reserve_call_record(
    sessionmaker: async_sessionmaker[AsyncSession],
    request: DialRequest,
    *,
    started_at: datetime,
) -> int:
    """Insert a fresh ``call_record`` row at dial time and return its id.

    The row is updated to terminal state by ``persist_call_record`` at END.
    We pre-create so any operator query (api / worker) can see the call as
    ``init`` while the engine is still running it.
    """

    async with sessionmaker() as db:
        record = CallRecord(
            lead_id=request.lead.lead_id,
            campaign_id=request.lead.campaign_id,
            caller_id=request.caller_id,
            status=CallStatus.INIT,
            started_at=started_at,
            transcript=[],
            prompt_versions=request.prompt_versions.model_dump(mode="json"),
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record.id


async def finalize_session(
    session: CallSession,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    redis: Redis[Any],
    settings: Settings,
    session_manager: SessionManager,
    started_at: datetime,
) -> None:
    """Run the post-call sequence: persist → LPUSH CallEnded → DECR → unregister.

    Each step is independent (DB / Redis live in different worlds, so we never
    bind them in the same transaction); failures are logged but the sequence
    always proceeds so concurrency counter cannot leak.
    """

    ended_at = now_utc()
    cause = _resolve_hangup_cause(session)
    session.hangup_cause = cause.value

    persisted = await persist_call_record(
        sessionmaker, session, started_at=started_at, ended_at=ended_at
    )
    if not persisted:
        logger.error(
            "finalize_persist_failed call_record_id=%s",
            session.call_record_id,
        )

    # Post-call extractor (pipeline-stream-and-referee): hand the transcript to
    # the worker's offline extractor. Only when an extractor slot is configured
    # and there is real dialog to extract from.
    if session.extractor_role_config_id and session.dialog_history:
        await _lpush_extract(
            redis, sessionmaker, session, max_retries=3
        )

    await _lpush_call_ended(
        redis,
        queue=settings.engine_call_ended_queue,
        call_record_id=session.call_record_id,
        hangup_cause=cause,
        ended_at=ended_at,
    )

    await _decr_concurrency(redis, key=settings.engine_concurrency_key)

    session_manager.unregister(session.call_record_id)
    logger.info(
        "session_finalized call_record_id=%s hangup_cause=%s persisted=%s",
        session.call_record_id,
        cause.value,
        persisted,
    )


def _resolve_hangup_cause(session: CallSession) -> HangupCause:
    """Map ``session.hangup_cause`` (string) to the typed enum, with a fallback."""

    raw = session.hangup_cause or HangupCause.NORMAL_CLEARING.value
    try:
        return HangupCause(raw)
    except ValueError:
        logger.warning(
            "unknown_hangup_cause_falling_back call_record_id=%s raw=%s",
            session.call_record_id,
            raw,
        )
        return HangupCause.NORMAL_CLEARING


async def _lpush_call_ended(
    redis: Redis[Any],
    *,
    queue: str,
    call_record_id: int,
    hangup_cause: HangupCause,
    ended_at: datetime,
    max_retries: int = 3,
) -> bool:
    msg = CallEnded(
        call_record_id=call_record_id,
        hangup_cause=hangup_cause,
        ended_at=ended_at,
    )
    payload = msg.model_dump_json()
    backoff = 0.2
    for attempt in range(1, max_retries + 1):
        try:
            await redis.lpush(queue, payload)
            return True
        except Exception:  # noqa: BLE001 — Redis transient failures are retried
            logger.exception(
                "call_ended_lpush_failed attempt=%s call_record_id=%s",
                attempt,
                call_record_id,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
    return False


async def _lpush_extract(
    redis: Redis[Any],
    sessionmaker: async_sessionmaker[AsyncSession],
    session: CallSession,
    *,
    max_retries: int = 3,
) -> bool:
    """LPUSH the post-call extract task + mark call_record.extract_status.

    Spec: service-communication § "isales:extract 队列消息 schema". The
    extract_status='pending' UPDATE is best-effort; a failed UPDATE does not
    block the LPUSH (the worker still writes 'done'/'failed' on completion).
    """
    payload = json.dumps(
        {
            "call_record_id": session.call_record_id,
            "transcript_snapshot": [
                {"role": t.role, "text": t.text, "ts_ms": t.ts_ms}
                for t in session.dialog_history
            ],
            "extractor_role_config_id": session.extractor_role_config_id,
            "extractor_prompt_version_id": session.extractor_prompt_version_id,
        },
        ensure_ascii=False,
    )

    pushed = False
    backoff = 0.2
    for attempt in range(1, max_retries + 1):
        try:
            await redis.lpush(EXTRACT_QUEUE, payload)
            pushed = True
            break
        except Exception:  # noqa: BLE001 — Redis transient failures are retried
            logger.exception(
                "extract_lpush_failed attempt=%s call_record_id=%s",
                attempt,
                session.call_record_id,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    if pushed:
        try:
            async with sessionmaker() as db:
                await db.execute(
                    update(CallRecord)
                    .where(CallRecord.id == session.call_record_id)
                    .values(extract_status="pending")
                )
                await db.commit()
        except Exception:  # noqa: BLE001 — don't block finalize on this UPDATE
            logger.exception(
                "extract_status_pending_update_failed call_record_id=%s",
                session.call_record_id,
            )
    return pushed


async def _decr_concurrency(
    redis: Redis[Any], *, key: str, max_retries: int = 3
) -> bool:
    backoff = 0.2
    for attempt in range(1, max_retries + 1):
        try:
            new = await redis.decr(key)
            if new < 0:
                # Snap back to zero — we DECRed past zero, meaning either
                # scheduler did not INCR for this call (mock / fake_dial path)
                # or another path leaked. Either way, don't leave it negative.
                await redis.set(key, 0)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("concurrency_decr_failed attempt=%s key=%s", attempt, key)
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
    return False
