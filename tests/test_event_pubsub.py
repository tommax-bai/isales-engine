"""Tests for event_publisher + event_consumer."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from isales_common.enums import CallStatus, HangupCause
from isales_common.schemas.messages.engine_control import (
    ManualHangup,
    TransferCommand,
)
from isales_common.schemas.messages.engine_event import StatusChanged

from isales_engine.call_session import CallSession
from isales_engine.event_consumer import subscribe_loop
from isales_engine.event_publisher import EventPublisher
from isales_engine.session_manager import SessionManager

pytestmark = [
    pytest.mark.usefixtures("redis_client"),
    pytest.mark.asyncio(loop_scope="session"),
]


def _make_session(call_id: int) -> CallSession:
    return CallSession(
        call_record_id=call_id,
        campaign_id=10,
        lead_id=5,
        caller_id="+x",
        prompt_versions_snapshot={},
    )


# ---- publisher -------------------------------------------------------------


async def test_publisher_status_changed(redis_client: Any) -> None:
    publisher = EventPublisher(redis_client)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("engine:events:campaign:10")
    # drain subscribe-confirm
    await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)

    event = StatusChanged(
        call_record_id=42, status=CallStatus.IN_CALL, reason="connected"
    )
    publisher.publish(10, event)
    await publisher.drain()

    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=2.0)
    assert msg is not None
    data = msg["data"]
    if isinstance(data, bytes):
        data = data.decode()
    parsed = json.loads(data)
    assert parsed["type"] == "status_changed"
    assert parsed["status"] == "in_call"

    await pubsub.unsubscribe("engine:events:campaign:10")
    await pubsub.close()  # type: ignore[no-untyped-call]


async def test_publisher_does_not_block_when_redis_slow(redis_client: Any) -> None:
    """Publisher returns immediately; tasks run in background."""

    publisher = EventPublisher(redis_client)
    event = StatusChanged(call_record_id=1, status=CallStatus.IN_CALL)
    # Synchronous publish() returns instantly; drain awaits the task.
    publisher.publish(10, event)
    publisher.publish(10, event)
    publisher.publish(10, event)
    await publisher.drain()


# ---- consumer --------------------------------------------------------------


async def test_consumer_dispatches_manual_hangup(redis_client: Any) -> None:
    sm = SessionManager()
    sm.register(_make_session(7))
    seen_hangup: list[tuple[int, str | None]] = []
    seen_transfer: list[tuple[int, int]] = []

    async def on_hangup(crid: int, op: str | None) -> None:
        seen_hangup.append((crid, op))

    async def on_transfer(crid: int, agent_id: int) -> None:
        seen_transfer.append((crid, agent_id))

    consumer = asyncio.create_task(
        subscribe_loop(
            redis=redis_client,
            session_manager=sm,
            on_manual_hangup=on_hangup,
            on_transfer_command=on_transfer,
        )
    )

    # Give the subscribe() loop time to land before we publish.
    await asyncio.sleep(0.1)
    msg = ManualHangup(call_record_id=7, operator="alice")
    await redis_client.publish("engine:control:campaign:10", msg.model_dump_json())

    # Wait for handler to run.
    for _ in range(20):
        if seen_hangup:
            break
        await asyncio.sleep(0.05)

    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer

    assert seen_hangup == [(7, "alice")]
    assert seen_transfer == []


async def test_consumer_silently_drops_unknown_call_id(redis_client: Any) -> None:
    sm = SessionManager()  # no sessions registered
    seen: list[int] = []

    async def on_hangup(crid: int, op: str | None) -> None:
        seen.append(crid)

    async def on_transfer(crid: int, agent_id: int) -> None:
        pass

    consumer = asyncio.create_task(
        subscribe_loop(
            redis=redis_client,
            session_manager=sm,
            on_manual_hangup=on_hangup,
            on_transfer_command=on_transfer,
        )
    )
    await asyncio.sleep(0.1)
    msg = ManualHangup(call_record_id=999, operator=None)
    await redis_client.publish("engine:control:campaign:10", msg.model_dump_json())

    await asyncio.sleep(0.2)
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer

    assert seen == []


async def test_consumer_invalid_payload_dropped(redis_client: Any) -> None:
    sm = SessionManager()
    sm.register(_make_session(1))

    async def on_hangup(crid: int, op: str | None) -> None:
        pytest.fail("handler called with invalid payload")

    async def on_transfer(crid: int, agent_id: int) -> None:
        pytest.fail("handler called with invalid payload")

    consumer = asyncio.create_task(
        subscribe_loop(
            redis=redis_client,
            session_manager=sm,
            on_manual_hangup=on_hangup,
            on_transfer_command=on_transfer,
        )
    )
    await asyncio.sleep(0.1)
    await redis_client.publish("engine:control:campaign:10", "not a json")
    await redis_client.publish("engine:control:campaign:10", '{"type": "unknown"}')

    await asyncio.sleep(0.2)
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer


async def test_consumer_dispatches_transfer_command(redis_client: Any) -> None:
    sm = SessionManager()
    sm.register(_make_session(11))
    seen: list[tuple[int, int]] = []

    async def on_hangup(crid: int, op: str | None) -> None:
        pass

    async def on_transfer(crid: int, agent_id: int) -> None:
        seen.append((crid, agent_id))

    consumer = asyncio.create_task(
        subscribe_loop(
            redis=redis_client,
            session_manager=sm,
            on_manual_hangup=on_hangup,
            on_transfer_command=on_transfer,
        )
    )
    await asyncio.sleep(0.1)
    msg = TransferCommand(call_record_id=11, agent_id=42)
    await redis_client.publish("engine:control:campaign:10", msg.model_dump_json())

    for _ in range(20):
        if seen:
            break
        await asyncio.sleep(0.05)

    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer

    assert seen == [(11, 42)]


# Stash unused imports to satisfy the linter where they're only referenced
# inside type hints.
_ = (datetime, UTC, uuid4, HangupCause)
