"""In-process telephony mock used by stage 4 tests.

Spec: design.md § 18 (MockTelephonyClient).

Behaviour:

* ``dial(call_id, phone)`` waits ``connect_delay_ms`` then emits ``connected``.
* ``hangup(call_id)`` emits ``local_hangup`` and closes the audio channels.
* ``simulate_remote_hangup(call_id)`` emits ``remote_hangup``.
* ``inject_user_audio(call_id, frames)`` feeds the caller→engine queue used by
  ``audio_in``.
* ``audio_out`` records every chunk to ``outbound_log[call_id]`` for assertions.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from isales_engine.realtime.telephony_client import (
    TelephonyClient,
    TelephonyEvent,
)

_SENTINEL: Any = object()


class MockTelephonyClient(TelephonyClient):
    def __init__(self, *, connect_delay_ms: int = 0) -> None:
        self._connect_delay_s = connect_delay_ms / 1000.0
        self._dialed_phones: dict[int, str] = {}
        self._events: dict[int, asyncio.Queue[TelephonyEvent | object]] = {}
        self._inbound: dict[int, asyncio.Queue[bytes | object]] = {}
        self.outbound_log: dict[int, list[bytes]] = {}
        self._connect_tasks: dict[int, asyncio.Task[None]] = {}

    # ---- internal helpers ----------------------------------------------

    def _ensure_call(self, call_id: int) -> None:
        if call_id not in self._events:
            self._events[call_id] = asyncio.Queue()
            self._inbound[call_id] = asyncio.Queue()
            self.outbound_log.setdefault(call_id, [])

    def _close_call(self, call_id: int) -> None:
        if call_id in self._inbound:
            self._inbound[call_id].put_nowait(_SENTINEL)
        if call_id in self._events:
            self._events[call_id].put_nowait(_SENTINEL)

    # ---- TelephonyClient -----------------------------------------------

    async def dial(self, call_id: int, phone: str) -> None:
        self._ensure_call(call_id)
        self._dialed_phones[call_id] = phone

        if self._connect_delay_s <= 0:
            # Inline emit so callers can immediately call hangup /
            # simulate_remote_hangup without an extra ``await asyncio.sleep(0)``.
            await self._events[call_id].put(
                TelephonyEvent(type="connected", call_id=call_id)
            )
            return

        async def _connect_after_delay() -> None:
            await asyncio.sleep(self._connect_delay_s)
            await self._events[call_id].put(
                TelephonyEvent(type="connected", call_id=call_id)
            )

        self._connect_tasks[call_id] = asyncio.create_task(_connect_after_delay())

    async def hangup(self, call_id: int) -> None:
        self._ensure_call(call_id)
        await self._events[call_id].put(
            TelephonyEvent(type="local_hangup", call_id=call_id)
        )
        self._close_call(call_id)

    async def simulate_remote_hangup(
        self, call_id: int, *, detail: str | None = None
    ) -> None:
        self._ensure_call(call_id)
        await self._events[call_id].put(
            TelephonyEvent(type="remote_hangup", call_id=call_id, detail=detail)
        )
        self._close_call(call_id)

    async def inject_user_audio(self, call_id: int, frames: list[bytes]) -> None:
        self._ensure_call(call_id)
        for frame in frames:
            await self._inbound[call_id].put(frame)

    async def audio_in(self, call_id: int) -> AsyncIterator[bytes]:
        self._ensure_call(call_id)
        q = self._inbound[call_id]
        while True:
            item = await q.get()
            if item is _SENTINEL:
                return
            assert isinstance(item, bytes)
            yield item

    async def audio_out(
        self, call_id: int, chunks: AsyncIterator[bytes]
    ) -> None:
        self._ensure_call(call_id)
        async for chunk in chunks:
            self.outbound_log[call_id].append(chunk)

    async def events(self, call_id: int) -> AsyncIterator[TelephonyEvent]:
        self._ensure_call(call_id)
        q = self._events[call_id]
        while True:
            item = await q.get()
            if item is _SENTINEL:
                return
            assert isinstance(item, TelephonyEvent)
            yield item
