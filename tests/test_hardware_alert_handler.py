"""Tests for the cloud-side hardware-alert logging handler.

Spec: arch-cloud-edge-split tasks.md § 9.3 (A2 ships structured logging
only; PG persistence deferred to D2 hardware-observability).
"""

from __future__ import annotations

import logging

import pytest
from isales_common.proto import cloud_edge_pb2 as pb
from isales_common.transport.cloud_edge import EdgeIdentity

from isales_engine.transport.hardware_alert_handler import (
    _payload_summary,
    log_hardware_alert,
)
from isales_engine.transport.session_dispatcher import EngineSessionDispatcher


def _identity(edge: str = "edge-1") -> EdgeIdentity:
    return EdgeIdentity(edge_device_id=edge)


def _alert_signal_lost(device_id: int = 1, signal: int = 5) -> pb.HardwareAlert:
    return pb.HardwareAlert(
        device_id=device_id, signal_lost=pb.SignalLost(last_signal_strength=signal)
    )


def _alert_sim_arrears(device_id: int = 1, balance: str = "0.5") -> pb.HardwareAlert:
    return pb.HardwareAlert(
        device_id=device_id, sim_arrears=pb.SimArrears(balance_text=balance)
    )


def _alert_modem_init_failed(device_id: int = 2) -> pb.HardwareAlert:
    return pb.HardwareAlert(
        device_id=device_id,
        modem_init_failed=pb.ModemInitFailed(stage="CCID", detail="timeout"),
    )


def _alert_audio_stalled() -> pb.HardwareAlert:
    return pb.HardwareAlert(
        device_id=0,
        audio_buffer_stalled=pb.AudioBufferStalled(
            direction="upstream", stalled_ms=350
        ),
    )


def _alert_sim_changed(device_id: int = 1, iccid: str = "8986") -> pb.HardwareAlert:
    return pb.HardwareAlert(
        device_id=device_id, sim_changed=pb.SimChanged(new_iccid=iccid)
    )


@pytest.mark.parametrize(
    "alert,expected_kind",
    [
        (_alert_signal_lost(), "signal_lost"),
        (_alert_sim_arrears(), "sim_arrears"),
        (_alert_modem_init_failed(), "modem_init_failed"),
        (_alert_audio_stalled(), "audio_buffer_stalled"),
        (_alert_sim_changed(), "sim_changed"),
    ],
)
def test_payload_summary_extracts_kind_specific_fields(
    alert: pb.HardwareAlert, expected_kind: str
) -> None:
    summary = _payload_summary(alert)
    assert summary["kind"] == expected_kind


def test_payload_summary_signal_lost_includes_strength() -> None:
    summary = _payload_summary(_alert_signal_lost(signal=12))
    assert summary["last_signal_strength"] == 12


def test_payload_summary_audio_stalled_includes_direction_and_ms() -> None:
    summary = _payload_summary(_alert_audio_stalled())
    assert summary["direction"] == "upstream"
    assert summary["stalled_ms"] == 350


@pytest.mark.asyncio
async def test_log_hardware_alert_emits_warning_with_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.WARNING, logger="isales_engine.transport.hardware_alert_handler"
    )
    await log_hardware_alert(_identity("edge-7"), _alert_signal_lost(device_id=3))

    records = [r for r in caplog.records if r.message == "hardware_alert"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.WARNING
    assert rec.edge_device_id == "edge-7"  # type: ignore[attr-defined]
    assert rec.device_id == 3  # type: ignore[attr-defined]
    assert rec.kind == "signal_lost"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_log_hardware_alert_dispatched_through_dispatcher(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The full dispatcher wire-up: handle_edge_message → on_hardware_alert
    → log_hardware_alert. Smoke-tests the cross-module contract.
    """
    caplog.set_level(
        logging.WARNING, logger="isales_engine.transport.hardware_alert_handler"
    )

    dispatcher = EngineSessionDispatcher()
    dispatcher.on_hardware_alert(log_hardware_alert)

    msg = pb.Edge2Cloud(hardware_alert=_alert_sim_arrears(device_id=4, balance="0"))
    await dispatcher.handle_edge_message(_identity("edge-9"), msg)

    records = [r for r in caplog.records if r.message == "hardware_alert"]
    assert len(records) == 1
    assert records[0].kind == "sim_arrears"  # type: ignore[attr-defined]
