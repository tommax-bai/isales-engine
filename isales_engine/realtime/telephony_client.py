"""Telephony client abstraction.

Spec: service-communication § 通信通道矩阵 (engine ↔ modem-controller row);
      device-hardware § (real implementation, stage 6).

Stage 4 ships :class:`MockTelephonyClient` (in ``mock_telephony``) which
simulates connect / remote-hangup / audio frames in-process. Stage 6 adds
``RealTelephonyClient`` over a Unix socket / WebSocket to the modem-controller
daemon.

Both implementations are exercised by ``tests/test_telephony_client_contract.py``
so regressions surface early in the impl-engine-hardware change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

TelephonyEventType = Literal[
    "connected",
    "local_hangup",
    "remote_hangup",
    "device_error",
]


@dataclass
class TelephonyEvent:
    type: TelephonyEventType
    call_id: int
    detail: str | None = None


class TelephonyClient(ABC):
    """Engine-side wrapper for the modem-controller dial/hangup/audio surface."""

    @abstractmethod
    async def dial(self, call_id: int, phone: str) -> None:
        """Initiate a dial. Implementations MUST eventually emit ``connected``
        (or a hangup event if dialing fails) on the ``events(call_id)`` stream."""

    @abstractmethod
    async def hangup(self, call_id: int) -> None:
        """Locally tear down the call. Emits ``local_hangup``."""

    @abstractmethod
    def audio_in(self, call_id: int) -> AsyncIterator[bytes]:
        """Caller→engine audio frames (drives ASR)."""

    @abstractmethod
    async def audio_out(self, call_id: int, chunks: AsyncIterator[bytes]) -> None:
        """Engine→caller audio (consume the iterator end-to-end before returning)."""

    @abstractmethod
    def events(self, call_id: int) -> AsyncIterator[TelephonyEvent]:
        """Per-call event stream — connected / hangup / device_error."""
