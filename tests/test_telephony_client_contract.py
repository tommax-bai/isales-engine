"""Contract tests for any TelephonyClient implementation.

Stage 4 only ships :class:`MockTelephonyClient`; stage 6's
``RealTelephonyClient`` (impl-engine-hardware change) MUST pass the same
suite. Each test parametrises over a builder so adding new implementations
in stage 6 is one fixture line.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.realtime.telephony_client import TelephonyClient

ClientFactory = Callable[[], TelephonyClient]


@pytest.fixture(params=[lambda: MockTelephonyClient(connect_delay_ms=0)])
def make_client(request) -> ClientFactory:  # type: ignore[no-untyped-def]
    return request.param


async def _read_one_event(client: TelephonyClient, call_id: int) -> str:
    async for ev in client.events(call_id):
        return ev.type
    return "no_event"


async def test_dial_then_connected_event(make_client: ClientFactory) -> None:
    client = make_client()
    await client.dial(call_id=1, phone="+8613800000000")
    typ = await _read_one_event(client, 1)
    assert typ == "connected"


async def test_local_hangup_emits_event_and_closes_channels(
    make_client: ClientFactory,
) -> None:
    client = make_client()
    await client.dial(call_id=1, phone="+x")
    await client.hangup(1)
    types: list[str] = []
    async for ev in client.events(1):
        types.append(ev.type)
    assert "connected" in types
    assert "local_hangup" in types
    # audio_in must terminate cleanly after hangup.
    frames: list[bytes] = []
    async for frame in client.audio_in(1):
        frames.append(frame)
    assert frames == []


async def test_remote_hangup_simulator(make_client: ClientFactory) -> None:
    client = make_client()
    if not isinstance(client, MockTelephonyClient):
        pytest.skip("only MockTelephonyClient ships simulate_remote_hangup")
    await client.dial(call_id=1, phone="+x")
    await client.simulate_remote_hangup(1, detail="user pressed end")
    types: list[str] = []
    details: list[str | None] = []
    async for ev in client.events(1):
        types.append(ev.type)
        details.append(ev.detail)
    assert "connected" in types
    assert "remote_hangup" in types
    assert "user pressed end" in details


async def test_audio_in_yields_injected_frames(make_client: ClientFactory) -> None:
    client = make_client()
    if not isinstance(client, MockTelephonyClient):
        pytest.skip("only MockTelephonyClient supports inject_user_audio")

    await client.dial(call_id=1, phone="+x")

    consumer_received: list[bytes] = []

    async def consume() -> None:
        async for frame in client.audio_in(1):
            consumer_received.append(frame)

    consumer = asyncio.create_task(consume())
    await client.inject_user_audio(1, [b"\x01\x02", b"\x03\x04"])
    # Let consumer drain.
    await asyncio.sleep(0)
    await client.hangup(1)
    await consumer

    assert consumer_received == [b"\x01\x02", b"\x03\x04"]


async def test_audio_out_records_chunks(make_client: ClientFactory) -> None:
    client = make_client()
    if not isinstance(client, MockTelephonyClient):
        pytest.skip("audio_out log is mock-specific")

    await client.dial(call_id=1, phone="+x")

    async def chunks() -> AsyncIterator[bytes]:
        for c in (b"a", b"b", b"c"):
            yield c

    await client.audio_out(1, chunks())
    assert client.outbound_log[1] == [b"a", b"b", b"c"]


async def test_dial_with_connect_delay_emits_connected_after_delay() -> None:
    client = MockTelephonyClient(connect_delay_ms=50)
    await client.dial(call_id=1, phone="+x")
    # Drain only the first event; it should arrive ~50ms later.
    start = asyncio.get_event_loop().time()
    typ = await _read_one_event(client, 1)
    elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
    assert typ == "connected"
    assert elapsed_ms >= 40  # slack for scheduler jitter
