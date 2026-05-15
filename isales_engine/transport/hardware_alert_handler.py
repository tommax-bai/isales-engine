"""Cloud-side hardware-alert handler.

Spec: arch-cloud-edge-split § device-hardware Requirement "modem-controller
      心跳与失联探测" (hardware alerts surface on the cloud-edge gRPC
      bidi stream as ``Edge2Cloud.HardwareAlert``);
      arch-cloud-edge-split tasks.md § 9.3.

A2 ships **structured logging only** per the task wording "A2 范围：仅落 PG
与基础日志告警；详细告警 / 一键诊断 由 D2 处理":

- Every HardwareAlert is logged at WARNING with the alert's kind +
  device_id + edge_device_id + payload, so operators can grep logs and
  ship-log alert rules can fire.
- PG persistence is intentionally deferred to D2
  ``hardware-observability`` which introduces the schema, retention
  policy, and boss-console aggregation view.

Wire-up at engine startup (Task 14 e2e demo)::

    dispatcher.on_hardware_alert(log_hardware_alert)
"""

from __future__ import annotations

import logging

from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity

logger = logging.getLogger(__name__)


def _payload_summary(alert: pb.HardwareAlert) -> dict[str, object]:
    """Extract the alert-kind-specific fields into a flat dict for logging.

    Returning a dict (not a pre-formatted string) keeps the structured
    logger happy and lets ops set log-rules on individual fields.
    """

    kind = alert.WhichOneof("kind")
    if kind == "signal_lost":
        return {
            "kind": "signal_lost",
            "last_signal_strength": alert.signal_lost.last_signal_strength,
        }
    if kind == "sim_arrears":
        return {
            "kind": "sim_arrears",
            "balance_text": alert.sim_arrears.balance_text,
        }
    if kind == "modem_init_failed":
        return {
            "kind": "modem_init_failed",
            "stage": alert.modem_init_failed.stage,
            "detail": alert.modem_init_failed.detail,
        }
    if kind == "audio_buffer_stalled":
        return {
            "kind": "audio_buffer_stalled",
            "direction": alert.audio_buffer_stalled.direction,
            "stalled_ms": alert.audio_buffer_stalled.stalled_ms,
        }
    if kind == "sim_changed":
        return {
            "kind": "sim_changed",
            "new_iccid": alert.sim_changed.new_iccid,
        }
    return {"kind": kind or "unknown"}


async def log_hardware_alert(
    identity: EdgeIdentity, alert: pb.HardwareAlert
) -> None:
    """Pass this directly to :meth:`EngineSessionDispatcher.on_hardware_alert`.

    Async signature matches :class:`HardwareAlertCallback` even though the
    handler is logging-only (D2 will replace this with a PG writer + boss-
    console fan-out).
    """

    ts = alert.ts.ToDatetime() if alert.HasField("ts") else None
    extras: dict[str, object] = {
        "edge_device_id": identity.edge_device_id,
        "device_id": alert.device_id,
        "ts": ts.isoformat() if ts is not None else None,
    }
    extras.update(_payload_summary(alert))
    logger.warning("hardware_alert", extra=extras)


__all__ = ["log_hardware_alert"]
