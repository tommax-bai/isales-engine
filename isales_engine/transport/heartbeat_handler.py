"""Cloud-side heartbeat handler — updates ``device.last_seen_at`` in PG.

Spec: arch-cloud-edge-split § device-hardware Requirement "modem-controller
      心跳与失联探测";
      arch-cloud-edge-split tasks.md § 9.4.

The cloud-edge bidi stream carries ``Edge2Cloud.Heartbeat`` every 30 s. Each
heartbeat includes a ``devices`` list of ``DeviceHealth`` rows summarizing
the modems attached to that edge. This handler writes each device's
``last_seen_at`` so the worker watchdog
(:mod:`isales_worker.device_watchdog`) can mark stale devices ``offline``
when the heartbeat goes silent for ≥ 120 s.

Wire-up at engine startup (Task 14 e2e demo)::

    handler = make_heartbeat_handler(sessionmaker)
    dispatcher.on_heartbeat(handler)

Idempotency: a heartbeat may arrive after a watchdog already flipped the
device offline; the handler only refreshes ``last_seen_at`` (it does NOT
change ``status``). Re-enabling a recovered device goes through the udev /
pyserial path that emits ``CallEvent.device_state_changed`` (spec
device-hardware Scenario "进程恢复").
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from isales_common.models import Device
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


HeartbeatHandler = Callable[[EdgeIdentity, pb.Heartbeat], Awaitable[None]]


def _heartbeat_ts(heartbeat: pb.Heartbeat) -> datetime:
    """Convert the Heartbeat's protobuf timestamp to a UTC datetime.

    Falls back to ``now()`` if the edge sent no ts (defensive — the proto
    always has it, but tests / legacy clients might not).
    """

    if heartbeat.HasField("ts"):
        ts: datetime = heartbeat.ts.ToDatetime(tzinfo=UTC)
        return ts
    return datetime.now(tz=UTC)


async def apply_heartbeat(
    session: AsyncSession,
    identity: EdgeIdentity,
    heartbeat: pb.Heartbeat,
) -> int:
    """Write ``last_seen_at`` for every device in the heartbeat.

    Returns the number of device rows touched (useful for tests + metrics).
    The caller MUST commit the session.
    """

    if not heartbeat.devices:
        # Cloud-side heartbeat echoes are device-less; edge heartbeats with
        # an empty list mean the edge has no modems attached yet. Both are
        # no-ops at the PG level (the stream's own liveness covers the
        # cloud-edge side; per-device staleness has nothing to update).
        return 0

    ts = _heartbeat_ts(heartbeat)
    touched = 0
    for device_health in heartbeat.devices:
        if device_health.device_id <= 0:
            logger.warning(
                "heartbeat_device_id_invalid",
                extra={
                    "edge_device_id": identity.edge_device_id,
                    "device_id": device_health.device_id,
                },
            )
            continue
        result = await session.execute(
            update(Device)
            .where(Device.id == device_health.device_id)
            .values(last_seen_at=ts)
        )
        rowcount = getattr(result, "rowcount", 0) or 0
        touched += int(rowcount)
    return touched


def make_heartbeat_handler(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> HeartbeatHandler:
    """Build a callback for :meth:`EngineSessionDispatcher.on_heartbeat`.

    Each invocation opens its own session, applies the heartbeat, commits.
    Exceptions are logged; the stream MUST stay open (transient PG issues
    shouldn't kill the heartbeat path).
    """

    async def _handler(identity: EdgeIdentity, heartbeat: pb.Heartbeat) -> None:
        try:
            async with sessionmaker() as session:
                touched = await apply_heartbeat(session, identity, heartbeat)
                await session.commit()
        except Exception:
            logger.exception(
                "heartbeat_handler_failed",
                extra={"edge_device_id": identity.edge_device_id},
            )
            return
        if touched:
            logger.debug(
                "heartbeat_applied",
                extra={
                    "edge_device_id": identity.edge_device_id,
                    "touched": touched,
                },
            )

    return _handler


__all__ = ["HeartbeatHandler", "apply_heartbeat", "make_heartbeat_handler"]
