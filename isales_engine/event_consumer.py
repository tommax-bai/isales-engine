"""EngineControl pub/sub consumer.

Spec: service-communication § 通信通道矩阵 (api→engine row);
      message-contract § Scenario "consumer 校验 schema_version".

Subscribes ``engine:control:campaign:*`` (pattern), validates
:class:`EngineControl` payloads, and dispatches to the live
``CallSession``. Unknown call_record_ids → WARN log, silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from isales_common.schemas.messages.engine_control import EngineControl
from pydantic import TypeAdapter, ValidationError
from redis.asyncio import Redis

from isales_engine.session_manager import SessionManager

logger = logging.getLogger(__name__)

PATTERN = "engine:control:campaign:*"

_control_adapter: TypeAdapter[EngineControl] = TypeAdapter(EngineControl)


HangupHandler = Callable[[int, str | None], Awaitable[None]]
"""(call_record_id, operator) -> None — drives the live session into END."""

TransferHandler = Callable[[int, int], Awaitable[None]]
"""(call_record_id, agent_id) -> None — kicks the live session into TRANSFERRING."""


async def subscribe_loop(
    *,
    redis: Redis[Any],
    session_manager: SessionManager,
    on_manual_hangup: HangupHandler,
    on_transfer_command: TransferHandler,
) -> None:
    """Long-running PSUBSCRIBE loop. Cancellation = graceful exit."""

    pubsub = redis.pubsub()
    await pubsub.psubscribe(PATTERN)
    try:
        while True:
            try:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("engine_control_get_message_error")
                await asyncio.sleep(1)
                continue

            if msg is None or msg.get("type") not in ("pmessage", "message"):
                continue

            data = msg.get("data")
            if isinstance(data, bytes):
                data = data.decode()
            if not isinstance(data, str):
                continue

            await _dispatch(
                data,
                session_manager=session_manager,
                on_manual_hangup=on_manual_hangup,
                on_transfer_command=on_transfer_command,
            )
    finally:
        try:
            await pubsub.punsubscribe(PATTERN)
            await pubsub.close()
        except Exception:
            logger.exception("engine_control_pubsub_close_error")


async def _dispatch(
    raw: str,
    *,
    session_manager: SessionManager,
    on_manual_hangup: HangupHandler,
    on_transfer_command: TransferHandler,
) -> None:
    try:
        msg = _control_adapter.validate_json(raw)
    except ValidationError as exc:
        logger.warning("engine_control_invalid errors=%s", exc.errors()[:3])
        return

    if session_manager.get(msg.call_record_id) is None:
        logger.warning(
            "engine_control_unknown_session type=%s call_record_id=%s",
            msg.type,
            msg.call_record_id,
        )
        return

    try:
        if msg.type == "manual_hangup":
            await on_manual_hangup(msg.call_record_id, msg.operator)
        elif msg.type == "transfer":
            await on_transfer_command(msg.call_record_id, msg.agent_id)
        else:
            logger.warning("engine_control_unknown_type type=%s", msg.type)
    except Exception:
        logger.exception(
            "engine_control_handler_error type=%s call_record_id=%s",
            msg.type,
            msg.call_record_id,
        )
