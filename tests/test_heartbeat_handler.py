"""Tests for the cloud-side heartbeat handler.

Spec: arch-cloud-edge-split § device-hardware "modem-controller 心跳与
失联探测"; tasks.md § 9.4.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from isales_common.enums import DeviceStatus
from isales_common.models import Device
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity
from sqlalchemy import select

from isales_engine.transport.heartbeat_handler import (
    apply_heartbeat,
    make_heartbeat_handler,
)
from isales_engine.transport.session_dispatcher import EngineSessionDispatcher


def _identity(edge: str = "edge-1") -> EdgeIdentity:
    return EdgeIdentity(edge_device_id=edge)


def _heartbeat(
    *,
    ts: datetime | None = None,
    devices: list[tuple[int, DeviceStatus]] | None = None,
) -> pb.Heartbeat:
    pb_ts = Timestamp()
    pb_ts.FromDatetime(ts or datetime.now(tz=UTC))
    hb = pb.Heartbeat(ts=pb_ts)
    for device_id, status in devices or []:
        hb.devices.add(
            device_id=device_id,
            status=getattr(pb, f"DEVICE_STATUS_{status.value.upper()}"),
            signal_strength=20,
        )
    return hb


async def _seed_device(sessionmaker_, *, name: str = "d1") -> int:  # type: ignore[no-untyped-def]
    async with sessionmaker_() as session:
        dev = Device(name=name, status=DeviceStatus.IDLE)
        session.add(dev)
        await session.commit()
        return dev.id


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_heartbeat_writes_last_seen_at(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    device_id = await _seed_device(sessionmaker_)
    ts = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
    hb = _heartbeat(ts=ts, devices=[(device_id, DeviceStatus.IDLE)])

    async with sessionmaker_() as session:
        touched = await apply_heartbeat(session, _identity(), hb)
        await session.commit()

    assert touched == 1
    async with sessionmaker_() as session:
        dev = (await session.execute(select(Device))).scalar_one()
        assert dev.last_seen_at == ts


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_heartbeat_syncs_status(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    """Heartbeat carrying a valid device_status updates PG device.status.

    This is the primary mechanism for recovering from edge crashes: the
    daemon restarts, sends heartbeat with IDLE, and the scheduler can
    pick the device again.
    """
    device_id = await _seed_device(sessionmaker_)

    # Manually set device to DIALING (simulating a crash mid-call)
    async with sessionmaker_() as session:
        dev = await session.get(Device, device_id)
        assert dev is not None
        dev.status = DeviceStatus.DIALING
        await session.commit()

    # Edge sends heartbeat with IDLE
    hb = _heartbeat(devices=[(device_id, DeviceStatus.IDLE)])
    async with sessionmaker_() as session:
        await apply_heartbeat(session, _identity(), hb)
        await session.commit()

    async with sessionmaker_() as session:
        dev = (await session.execute(select(Device))).scalar_one()
        assert dev.status == DeviceStatus.IDLE


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_heartbeat_with_empty_devices_is_noop(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    await _seed_device(sessionmaker_)
    hb = _heartbeat()

    async with sessionmaker_() as session:
        touched = await apply_heartbeat(session, _identity(), hb)
        await session.commit()
    assert touched == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_heartbeat_skips_unknown_device_id(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    # No device row, but heartbeat references device_id=999 → update affects 0
    hb = _heartbeat(devices=[(999, DeviceStatus.IDLE)])
    async with sessionmaker_() as session:
        touched = await apply_heartbeat(session, _identity(), hb)
        await session.commit()
    assert touched == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_heartbeat_skips_invalid_device_id(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    hb = _heartbeat(devices=[(0, DeviceStatus.IDLE)])
    async with sessionmaker_() as session:
        touched = await apply_heartbeat(session, _identity(), hb)
        await session.commit()
    assert touched == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_heartbeat_handler_wired_through_dispatcher(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    device_id = await _seed_device(sessionmaker_)
    dispatcher = EngineSessionDispatcher()
    dispatcher.on_heartbeat(make_heartbeat_handler(sessionmaker_))

    ts = datetime.now(tz=UTC) - timedelta(seconds=10)
    msg = pb.Edge2Cloud(
        heartbeat=_heartbeat(ts=ts, devices=[(device_id, DeviceStatus.IDLE)])
    )
    await dispatcher.handle_edge_message(_identity(), msg)

    async with sessionmaker_() as session:
        dev = (await session.execute(select(Device))).scalar_one()
        # Compare within 1s tolerance (Timestamp precision)
        assert dev.last_seen_at is not None
        delta = abs((dev.last_seen_at - ts).total_seconds())
        assert delta < 1.0
