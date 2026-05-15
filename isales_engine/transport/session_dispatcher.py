"""Per-call routing of inbound cloud-edge messages.

Spec: arch-cloud-edge-split / design.md Decision 5 (cancel routed in-
process) + architecture spec § Requirement: Engine session 与 cloud
instance co-location.

The :class:`CloudEdgeGrpcServer` exposes a single ``on_edge_message``
callback for ALL inbound frames from ALL connected edges. The engine
needs to fan that out: a CallEvent for ``call_id="c-1"`` belongs to the
engine session that issued the dial for ``c-1``; a HardwareAlert isn't
call-scoped at all and goes to the worker for persistence.

:class:`EngineSessionDispatcher` is that fan-out. Sessions register
their callbacks against a ``call_id`` before issuing the dial; the
dispatcher consults the registry every time a frame arrives and
delivers in-process via ``await``. Redis Pub/Sub is NOT in this path —
that's the whole point of the "cancel not via Redis" decision.

Identity isolation: every dispatched frame is checked against the
``edge_device_id`` the session was registered with. A frame for a call
that arrives from a *different* edge is dropped — this is a
defense-in-depth guard against cross-tenant routing once C2's
multi-tenancy ships (an edge can only see calls it owns).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity

logger = logging.getLogger(__name__)


CallEventCallback: TypeAlias = Callable[[pb.CallEvent], Awaitable[None]]
DialAckCallback: TypeAlias = Callable[[pb.DialAck], Awaitable[None]]
HardwareAlertCallback: TypeAlias = Callable[[EdgeIdentity, pb.HardwareAlert], Awaitable[None]]
HeartbeatCallback: TypeAlias = Callable[[EdgeIdentity, pb.Heartbeat], Awaitable[None]]


@dataclass
class _SessionEntry:
    """One registered session's binding."""

    call_id: str
    edge_device_id: str
    on_call_event: CallEventCallback
    on_dial_ack: DialAckCallback | None


class EngineSessionDispatcher:
    """Routes inbound Edge2Cloud frames to the right active engine session.

    Wire-up at engine startup::

        dispatcher = EngineSessionDispatcher()
        dispatcher.on_hardware_alert(worker_alert_handler)
        grpc_server.on_edge_message(dispatcher.handle_edge_message)

    Per-call usage::

        dispatcher.register(
            call_id="c-1",
            edge_device_id="edge-7",
            on_call_event=session.on_call_event,
            on_dial_ack=session.on_dial_ack,
        )
        try:
            await grpc_server.send_to_edge("edge-7", dial_command_for_c1)
            # ...session.run()...
        finally:
            dispatcher.deregister("c-1")

    Thread safety: methods are all sync or async; the registry is
    accessed only from the asyncio event loop where ``handle_edge_message``
    is invoked. Sessions register / deregister on that same loop, so no
    explicit lock is required.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionEntry] = {}
        self._hardware_alert_cb: HardwareAlertCallback | None = None
        self._heartbeat_cb: HeartbeatCallback | None = None

    # ----- session registry -----------------------------------------------

    def register(
        self,
        call_id: str,
        *,
        edge_device_id: str,
        on_call_event: CallEventCallback,
        on_dial_ack: DialAckCallback | None = None,
    ) -> None:
        """Bind a session's callbacks to a ``call_id``.

        Synchronous so the caller can register BEFORE awaiting
        :meth:`send_to_edge` for the DialCommand — otherwise a fast DialAck
        or CallEvent could arrive at the dispatcher before the binding
        exists and get logged as "unknown call_id".

        If ``call_id`` is already bound, raises :class:`ValueError`
        (rather than silently displacing — call_ids are server-generated
        and reuse implies a bug upstream).
        """
        if call_id in self._sessions:
            raise ValueError(f"call_id already registered: {call_id!r}")
        self._sessions[call_id] = _SessionEntry(
            call_id=call_id,
            edge_device_id=edge_device_id,
            on_call_event=on_call_event,
            on_dial_ack=on_dial_ack,
        )

    def deregister(self, call_id: str) -> None:
        """Remove the binding. Idempotent — deregistering an unknown
        ``call_id`` is a no-op (the session may have already exited)."""
        self._sessions.pop(call_id, None)

    def is_registered(self, call_id: str) -> bool:
        """Test helper / introspection. Production code should not depend
        on dispatcher state directly."""
        return call_id in self._sessions

    def on_hardware_alert(self, callback: HardwareAlertCallback) -> None:
        """Register the process-wide hardware-alert handler.

        HardwareAlert frames are not bound to any call_id — modem signal
        loss / SIM arrears / etc. flow to the worker for persistence and
        aggregation, not to a specific engine session. Only one handler
        is supported; a second call replaces the first.
        """
        self._hardware_alert_cb = callback

    def on_heartbeat(self, callback: HeartbeatCallback) -> None:
        """Register the process-wide heartbeat handler.

        Heartbeat frames carry per-device liveness summaries (see
        arch-cloud-edge-split § device-hardware "modem-controller 心跳与失联
        探测"). The handler is responsible for updating
        ``device.last_seen_at`` so the worker watchdog can flip stale
        devices to ``offline``. Only one handler is supported; a second
        call replaces the first. Missing handler is a hard no-op (early
        bring-up before task 9.4 wires this).
        """
        self._heartbeat_cb = callback

    # ----- inbound dispatch ----------------------------------------------

    async def handle_edge_message(
        self,
        identity: EdgeIdentity,
        msg: pb.Edge2Cloud,
    ) -> None:
        """Main entry point — pass this to ``grpc_server.on_edge_message``.

        Dispatches the oneof payload to the appropriate handler. Unknown
        or empty payloads are logged at WARNING and dropped (do not raise
        — the gRPC server already protects the stream from handler
        exceptions, but warnings here surface upstream protocol drift).
        """
        kind = msg.WhichOneof("payload")
        if kind == "call_event":
            await self._dispatch_call_event(identity, msg.call_event)
        elif kind == "dial_ack":
            await self._dispatch_dial_ack(identity, msg.dial_ack)
        elif kind == "hardware_alert":
            await self._dispatch_hardware_alert(identity, msg.hardware_alert)
        elif kind == "heartbeat":
            await self._dispatch_heartbeat(identity, msg.heartbeat)
        elif kind is None:
            logger.warning(
                "edge_message with empty payload",
                extra={"edge_device_id": identity.edge_device_id},
            )
        else:
            logger.warning(
                "edge_message with unknown payload kind",
                extra={
                    "edge_device_id": identity.edge_device_id,
                    "kind": kind,
                },
            )

    async def _dispatch_call_event(
        self,
        identity: EdgeIdentity,
        event: pb.CallEvent,
    ) -> None:
        entry = self._sessions.get(event.call_id)
        if entry is None:
            # Late events arrive after deregister (e.g. the session hung
            # up locally and the edge's remote_hangup raced). Not an
            # error condition — log at INFO and drop.
            logger.info(
                "call_event for unknown call_id; possibly late event after deregister",
                extra={
                    "edge_device_id": identity.edge_device_id,
                    "call_id": event.call_id,
                    "kind": event.WhichOneof("kind"),
                },
            )
            return
        if entry.edge_device_id != identity.edge_device_id:
            logger.warning(
                "call_event identity mismatch; dropping",
                extra={
                    "call_id": event.call_id,
                    "frame_edge": identity.edge_device_id,
                    "session_edge": entry.edge_device_id,
                },
            )
            return
        await entry.on_call_event(event)

    async def _dispatch_dial_ack(
        self,
        identity: EdgeIdentity,
        ack: pb.DialAck,
    ) -> None:
        entry = self._sessions.get(ack.call_id)
        if entry is None or entry.on_dial_ack is None:
            return
        if entry.edge_device_id != identity.edge_device_id:
            logger.warning(
                "dial_ack identity mismatch; dropping",
                extra={
                    "call_id": ack.call_id,
                    "frame_edge": identity.edge_device_id,
                    "session_edge": entry.edge_device_id,
                },
            )
            return
        await entry.on_dial_ack(ack)

    async def _dispatch_hardware_alert(
        self,
        identity: EdgeIdentity,
        alert: pb.HardwareAlert,
    ) -> None:
        if self._hardware_alert_cb is None:
            # Worker not wired up yet; alerts go to /dev/null. Log at
            # DEBUG — this is normal during partial bring-up.
            logger.debug(
                "hardware_alert dropped — no handler registered",
                extra={"edge_device_id": identity.edge_device_id},
            )
            return
        await self._hardware_alert_cb(identity, alert)

    async def _dispatch_heartbeat(
        self,
        identity: EdgeIdentity,
        heartbeat: pb.Heartbeat,
    ) -> None:
        if self._heartbeat_cb is None:
            # Liveness is the transport layer's concern when no handler
            # is wired — the gRPC bidi keeps itself alive regardless.
            return
        await self._heartbeat_cb(identity, heartbeat)


__all__ = [
    "CallEventCallback",
    "DialAckCallback",
    "EngineSessionDispatcher",
    "HardwareAlertCallback",
    "HeartbeatCallback",
]
