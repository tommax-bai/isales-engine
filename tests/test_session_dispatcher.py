"""EngineSessionDispatcher routing tests.

Spec: arch-cloud-edge-split / design.md Decision 5 + architecture spec
§ Engine session co-location.

These tests exercise the dispatcher in isolation — no gRPC server, no
real edge connection. They focus on the routing contract:

- CallEvent / DialAck route to the per-call callbacks bound by register().
- HardwareAlert routes to the process-wide handler.
- Late frames after deregister are logged and dropped (no exception).
- Identity-mismatch frames are dropped (cross-edge isolation).
"""

from __future__ import annotations

import logging

import pytest
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity

from isales_engine.transport.session_dispatcher import EngineSessionDispatcher


def _identity(edge_id: str = "edge-1") -> EdgeIdentity:
    return EdgeIdentity(edge_device_id=edge_id)


# --------------------------------------------------------------------------
# register / deregister
# --------------------------------------------------------------------------


async def _noop(_: object) -> None:
    pass


def test_register_then_is_registered() -> None:
    dispatcher = EngineSessionDispatcher()
    assert not dispatcher.is_registered("c-1")
    dispatcher.register("c-1", edge_device_id="edge-1", on_call_event=_noop)
    assert dispatcher.is_registered("c-1")


def test_register_duplicate_call_id_raises() -> None:
    """Duplicate call_ids imply a bug upstream — fail loudly rather than
    silently displacing the existing binding."""
    dispatcher = EngineSessionDispatcher()
    dispatcher.register("c-1", edge_device_id="edge-1", on_call_event=_noop)
    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register("c-1", edge_device_id="edge-2", on_call_event=_noop)


def test_deregister_is_idempotent() -> None:
    dispatcher = EngineSessionDispatcher()
    dispatcher.register("c-1", edge_device_id="edge-1", on_call_event=_noop)
    dispatcher.deregister("c-1")
    dispatcher.deregister("c-1")  # no exception
    dispatcher.deregister("never-registered")  # no exception
    assert not dispatcher.is_registered("c-1")


# --------------------------------------------------------------------------
# CallEvent routing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_event_routes_to_bound_callback() -> None:
    received: list[pb.CallEvent] = []

    async def cb(event: pb.CallEvent) -> None:
        received.append(event)

    dispatcher = EngineSessionDispatcher()
    dispatcher.register("c-1", edge_device_id="edge-1", on_call_event=cb)

    event = pb.CallEvent(call_id="c-1", connected=pb.Connected())
    await dispatcher.handle_edge_message(
        _identity("edge-1"),
        pb.Edge2Cloud(call_event=event),
    )

    assert len(received) == 1
    assert received[0].WhichOneof("kind") == "connected"


@pytest.mark.asyncio
async def test_call_event_for_unknown_call_id_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Late events arriving after deregister are normal (the session may
    have exited locally before the edge's hangup reached us). Log at INFO
    and drop — do not raise, do not crash the stream."""
    dispatcher = EngineSessionDispatcher()
    with caplog.at_level(logging.INFO):
        await dispatcher.handle_edge_message(
            _identity(),
            pb.Edge2Cloud(
                call_event=pb.CallEvent(call_id="ghost", connected=pb.Connected()),
            ),
        )
    assert any("unknown call_id" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_call_event_identity_mismatch_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame for a registered call but from the wrong edge MUST NOT
    be delivered. This is defence-in-depth against cross-tenant routing
    once C2's multi-tenancy ships."""
    called = False

    async def cb(_event: pb.CallEvent) -> None:
        nonlocal called
        called = True

    dispatcher = EngineSessionDispatcher()
    dispatcher.register("c-1", edge_device_id="edge-A", on_call_event=cb)

    with caplog.at_level(logging.WARNING):
        await dispatcher.handle_edge_message(
            _identity("edge-B"),  # WRONG edge
            pb.Edge2Cloud(
                call_event=pb.CallEvent(call_id="c-1", connected=pb.Connected()),
            ),
        )

    assert called is False
    assert any("identity mismatch" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# DialAck routing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dial_ack_routes_to_bound_callback() -> None:
    received: list[pb.DialAck] = []

    async def on_ack(ack: pb.DialAck) -> None:
        received.append(ack)

    async def on_event(_event: pb.CallEvent) -> None:
        pass

    dispatcher = EngineSessionDispatcher()
    dispatcher.register(
        "c-1",
        edge_device_id="edge-1",
        on_call_event=on_event,
        on_dial_ack=on_ack,
    )

    await dispatcher.handle_edge_message(
        _identity("edge-1"),
        pb.Edge2Cloud(dial_ack=pb.DialAck(call_id="c-1", accepted=True)),
    )

    assert len(received) == 1
    assert received[0].accepted is True


@pytest.mark.asyncio
async def test_dial_ack_with_no_callback_is_silently_dropped() -> None:
    """A session that doesn't care about DialAck (just waits for the
    eventual CallEvent.connected) should be able to omit on_dial_ack
    without the dispatcher raising."""
    dispatcher = EngineSessionDispatcher()
    dispatcher.register("c-1", edge_device_id="edge-1", on_call_event=_noop)

    # Should not raise — exercise the path.
    await dispatcher.handle_edge_message(
        _identity("edge-1"),
        pb.Edge2Cloud(dial_ack=pb.DialAck(call_id="c-1", accepted=True)),
    )


# --------------------------------------------------------------------------
# HardwareAlert routing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hardware_alert_routes_to_process_wide_handler() -> None:
    received: list[tuple[EdgeIdentity, pb.HardwareAlert]] = []

    async def handler(identity: EdgeIdentity, alert: pb.HardwareAlert) -> None:
        received.append((identity, alert))

    dispatcher = EngineSessionDispatcher()
    dispatcher.on_hardware_alert(handler)

    alert = pb.HardwareAlert(
        device_id=7,
        signal_lost=pb.SignalLost(last_signal_strength=2),
    )
    await dispatcher.handle_edge_message(
        _identity("edge-1"),
        pb.Edge2Cloud(hardware_alert=alert),
    )

    assert len(received) == 1
    identity, frame_alert = received[0]
    assert identity.edge_device_id == "edge-1"
    assert frame_alert.WhichOneof("kind") == "signal_lost"


@pytest.mark.asyncio
async def test_hardware_alert_with_no_handler_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """During partial bring-up the worker may not be wired yet; alerts
    should drop silently at DEBUG rather than raise."""
    dispatcher = EngineSessionDispatcher()
    with caplog.at_level(logging.DEBUG):
        await dispatcher.handle_edge_message(
            _identity(),
            pb.Edge2Cloud(
                hardware_alert=pb.HardwareAlert(
                    sim_arrears=pb.SimArrears(balance_text="0.00"),
                ),
            ),
        )
    assert any("no handler registered" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Heartbeat + unknown payload
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_is_noop() -> None:
    """Heartbeats are the transport layer's concern; the dispatcher has
    nothing to route per-call. Just don't crash."""
    dispatcher = EngineSessionDispatcher()
    await dispatcher.handle_edge_message(
        _identity(),
        pb.Edge2Cloud(heartbeat=pb.Heartbeat()),
    )


@pytest.mark.asyncio
async def test_empty_payload_logged_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An Edge2Cloud envelope with no oneof variant set means upstream
    is shipping a malformed frame. Warn so we notice in production logs,
    but don't crash."""
    dispatcher = EngineSessionDispatcher()
    with caplog.at_level(logging.WARNING):
        await dispatcher.handle_edge_message(
            _identity(),
            pb.Edge2Cloud(),
        )
    assert any("empty payload" in r.message for r in caplog.records)
