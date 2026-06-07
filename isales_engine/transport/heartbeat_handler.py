"""Cloud-side heartbeat handler — updates ``device.last_seen_at`` and
``device.status`` in PG.

Spec: arch-cloud-edge-split § device-hardware Requirement "modem-controller
      心跳与失联探测";
      arch-cloud-edge-split tasks.md § 9.4.

The cloud-edge bidi stream carries ``Edge2Cloud.Heartbeat`` every 30 s. Each
heartbeat includes a ``devices`` list of ``DeviceHealth`` rows summarizing
the modems attached to that edge. This handler writes each device's
``last_seen_at`` and syncs ``status`` so the scheduler's
``pick_idle_device()`` sees the correct state even after an edge daemon
restart (e.g. recovering from a crash that left ``status='dialing'``).

Wire-up at engine startup (Task 14 e2e demo)::

    handler = make_heartbeat_handler(sessionmaker)
    dispatcher.on_heartbeat(handler)

Idempotency: receiving the same heartbeat status multiple times is a no-op
(the UPDATE is unconditional but the value is the same). The handler is
the single source of truth for edge-reported device state.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from isales_common.enums import DeviceStatus
from isales_common.models import Device
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


HeartbeatHandler = Callable[[EdgeIdentity, pb.Heartbeat], Awaitable[None]]

# Mapping from proto DeviceStatus enum to Python DeviceStatus StrEnum.
# DEVICE_STATUS_UNSPECIFIED (0) is excluded — it means "not set" and should
# not trigger a status update. DEVICE_STATUS_ERROR (9) has no Python
# counterpart; map to OFFLINE as a safe fallback.
_PROTO_STATUS_TO_ENUM: dict[int, DeviceStatus] = {
    int(pb.DEVICE_STATUS_UNKNOWN): DeviceStatus.UNKNOWN,
    int(pb.DEVICE_STATUS_DETECTED): DeviceStatus.DETECTED,
    int(pb.DEVICE_STATUS_REGISTERED): DeviceStatus.REGISTERED,
    int(pb.DEVICE_STATUS_IDLE): DeviceStatus.IDLE,
    int(pb.DEVICE_STATUS_DIALING): DeviceStatus.DIALING,
    int(pb.DEVICE_STATUS_IN_CALL): DeviceStatus.IN_CALL,
    int(pb.DEVICE_STATUS_OFFLINE): DeviceStatus.OFFLINE,
    int(pb.DEVICE_STATUS_FLAGGED): DeviceStatus.FLAGGED,
    int(pb.DEVICE_STATUS_ERROR): DeviceStatus.OFFLINE,
}


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

        # Build the values dict: always update last_seen_at; conditionally
        # sync status when the edge reports a meaningful value.
        values: dict[str, object] = {"last_seen_at": ts}
        proto_status = int(device_health.status)
        mapped_status = _PROTO_STATUS_TO_ENUM.get(proto_status)
        if mapped_status is not None:
            values["status"] = mapped_status

        result = await session.execute(
            update(Device)
            .where(Device.id == device_health.device_id)
            .values(**values)
        )
        rowcount = getattr(result, "rowcount", 0) or 0
        if rowcount and mapped_status is not None:
            logger.info(
                "device_status_synced device_id=%d status=%s",
                device_health.device_id,
                mapped_status.value,
            )
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
