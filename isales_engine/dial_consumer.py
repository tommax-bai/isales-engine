"""DialRequest queue consumer — engine entry point.

Spec: service-communication § 通信通道矩阵 (scheduler→engine row);
      message-contract § Scenario "consumer 校验 schema_version".

Reads ``engine:dial`` (BLPOP, 1s timeout), validates the message against the
isales-common ``DialRequest`` schema, reserves a ``call_record`` row, builds
a :class:`CallSession`, registers with :class:`SessionManager`, and spawns a
fire-and-forget per-call task that runs ``session_runner(session)`` followed
by ``finalize_session``.

PR #11 wires the real session runner (state machine + pipeline). PR #3 keeps
the runner pluggable so dial-side concerns (DLQ, schema validation,
call_record reservation, finalize wiring) can be tested in isolation.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from isales_common.schemas.messages.dial import DialRequest
from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from isales_engine.call_lifecycle import finalize_session, reserve_call_record
from isales_engine.call_session import CallSession
from isales_engine.session_manager import SessionManager
from isales_engine.settings import Settings
from isales_engine.transcript_recorder import now_utc

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = {1}

SessionRunner = Callable[[CallSession], Awaitable[None]]


def _peek_schema_version(raw: str) -> int | None:
    try:
        parsed = _json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    sv = parsed.get("schema_version")
    return sv if isinstance(sv, int) else None


async def handle_dial(
    raw: str,
    *,
    redis: Redis[Any],
    sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
    session_manager: SessionManager,
    runner: SessionRunner,
) -> CallSession | None:
    """Validate one raw message and spawn a session task.

    Returns the spawned ``CallSession`` (so tests can assert on it) or ``None``
    on DLQ / validation failure.
    """

    sv = _peek_schema_version(raw)
    if sv is None or sv not in SUPPORTED_SCHEMA_VERSIONS:
        logger.warning("dial_dlq schema_version=%s raw_len=%d", sv, len(raw))
        await redis.lpush(settings.engine_dlq, raw)
        return None

    try:
        request = DialRequest.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("dial_dlq validation errors=%s", exc.errors()[:3])
        await redis.lpush(settings.engine_dlq, raw)
        return None

    started_at = now_utc()
    try:
        call_record_id = await reserve_call_record(
            sessionmaker, request, started_at=started_at
        )
    except Exception:
        logger.exception(
            "reserve_call_record_failed lead_id=%s campaign_id=%s",
            request.lead.lead_id,
            request.lead.campaign_id,
        )
        await redis.lpush(settings.engine_dlq, raw)
        return None

    session = CallSession(
        call_record_id=call_record_id,
        campaign_id=request.lead.campaign_id,
        lead_id=request.lead.lead_id,
        caller_id=request.caller_id,
        prompt_versions_snapshot=request.prompt_versions.model_dump(mode="json"),
        device_id=request.device_id,
    )
    session_manager.register(session)

    logger.info(
        "dial_received call_record_id=%s campaign_id=%s lead_id=%s message_id=%s",
        call_record_id,
        session.campaign_id,
        session.lead_id,
        request.message_id,
    )

    task = asyncio.create_task(
        _runner_with_finalize(
            session,
            runner=runner,
            sessionmaker=sessionmaker,
            redis=redis,
            settings=settings,
            session_manager=session_manager,
            started_at=started_at,
        ),
        name=f"call_session:{call_record_id}",
    )
    session.tasks["main"] = task
    return session


async def _runner_with_finalize(
    session: CallSession,
    *,
    runner: SessionRunner,
    sessionmaker: async_sessionmaker[AsyncSession],
    redis: Redis[Any],
    settings: Settings,
    session_manager: SessionManager,
    started_at: Any,
) -> None:
    """Run the session and unconditionally finalize.

    Cancellation handling: if the runner is cancelled (graceful shutdown),
    set a sentinel hangup cause and still finalize. We use ``asyncio.shield``
    around finalize so the post-call writes complete even if the parent
    cancellation is still propagating.
    """

    cancelled = False
    try:
        await runner(session)
    except asyncio.CancelledError:
        cancelled = True
        if session.hangup_cause is None:
            session.hangup_cause = "manual_hangup"
    except Exception:
        logger.exception(
            "session_runner_error call_record_id=%s", session.call_record_id
        )
        if session.hangup_cause is None:
            session.hangup_cause = "no_progress_timeout"

    try:
        await asyncio.shield(
            finalize_session(
                session,
                sessionmaker=sessionmaker,
                redis=redis,
                settings=settings,
                session_manager=session_manager,
                started_at=started_at,
            )
        )
    except Exception:
        logger.exception(
            "session_finalize_error call_record_id=%s", session.call_record_id
        )

    if cancelled:
        raise asyncio.CancelledError()


async def dial_loop(
    *,
    redis: Redis[Any],
    sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
    session_manager: SessionManager,
    runner: SessionRunner,
) -> None:
    """Long-running BLPOP loop. Cancellation = graceful exit."""

    while True:
        try:
            popped = await redis.blpop([settings.engine_dial_queue], timeout=1)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dial_blpop_error")
            await asyncio.sleep(1)
            continue

        if popped is None:
            continue
        _key, raw = popped
        try:
            await handle_dial(
                raw,
                redis=redis,
                sessionmaker=sessionmaker,
                settings=settings,
                session_manager=session_manager,
                runner=runner,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dial_handle_error raw_len=%d", len(raw))
