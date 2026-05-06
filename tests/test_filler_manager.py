"""Tests for FillerManager (filler spec)."""

from __future__ import annotations

import asyncio

import pytest

from isales_engine.call_session import CallSession
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.filler_manager import (
    FillerManager,
    FillerPhraseSpec,
    FillerSetSpec,
)
from isales_engine.realtime.mock_telephony import MockTelephonyClient


def _session(call_id: int = 1) -> CallSession:
    s = CallSession(
        call_record_id=call_id,
        campaign_id=10,
        lead_id=5,
        caller_id="+x",
        prompt_versions_snapshot={},
    )
    return s


def _set(set_id: int, *phrases: tuple[int, str, str]) -> FillerSetSpec:
    """Convenience: each tuple is (phrase_id, text, status)."""

    return FillerSetSpec(
        id=set_id,
        sort_order=set_id,
        phrases=[
            FillerPhraseSpec(
                id=pid,
                text=text,
                audio_url=f"oss://x/{pid}.wav" if status == "ready" else None,
                generation_status=status,
            )
            for pid, text, status in phrases
        ],
    )


async def _new_mock_telephony(call_id: int = 1) -> MockTelephonyClient:
    client = MockTelephonyClient(connect_delay_ms=0)
    await client.dial(call_id, "+x")
    # drain the connected event so the queue is clean
    async for _ in client.events(call_id):
        break
    return client


# ---- selection -------------------------------------------------------------


async def test_filler_skip_when_no_ready_phrase() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    sets = [_set(1, (1, "hi", "pending"))]
    fm = FillerManager(session, sets, telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()
    assert tel.outbound_log[1] == []
    assert all(e["type"] != "filler" for e in session.full_transcript)


async def test_filler_picks_ready_phrase_and_writes_event() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    sets = [_set(1, (1, "嗯", "ready"), (2, "稍等", "ready"))]
    fm = FillerManager(session, sets, telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()

    out = tel.outbound_log[1]
    assert len(out) > 0  # PCM bytes were forwarded
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1
    assert fillers[0]["filler_phrase_id"] in {1, 2}


async def test_multi_set_round_robin_and_dedup() -> None:
    """3 sets × 2 phrases each, 7 turns. Ensure round-robin walk + per-set
    dedup, and that exhausted sets reset."""

    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=2, chunk_size=4)
    sets = [
        _set(1, (10, "a1", "ready"), (11, "a2", "ready")),
        _set(2, (20, "b1", "ready"), (21, "b2", "ready")),
        _set(3, (30, "c1", "ready"), (31, "c2", "ready")),
    ]

    chosen_sets: list[int] = []
    chosen_phrases: list[int] = []
    for _ in range(7):
        fm = FillerManager(session, sets, telephony=tel, tts=tts)
        await fm.start()
        await fm.wait_finished()
        last = next(
            e for e in reversed(session.full_transcript) if e["type"] == "filler"
        )
        pid = last["filler_phrase_id"]
        # Map pid back to its set.
        set_id = next(s.id for s in sets if any(p.id == pid for p in s.phrases))
        chosen_sets.append(set_id)
        chosen_phrases.append(pid)

    # Round-robin order across 7 turns: 1, 2, 3, 1, 2, 3, 1.
    assert chosen_sets == [1, 2, 3, 1, 2, 3, 1]

    # Per-set dedup: turns 1 and 4 land on set 1 with two distinct phrases.
    set1_phrases = [chosen_phrases[i] for i, sid in enumerate(chosen_sets) if sid == 1]
    assert len(set1_phrases) == 3
    assert set1_phrases[0] != set1_phrases[1]  # first two trips different
    # On the third trip the per-set "used" should reset → could repeat any.


async def test_stop_cancels_playback_and_skips_event() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=320, chunk_size=320, chunk_delay_s=0.05)
    sets = [_set(1, (1, "abcdefghijklmnopqrst", "ready"))]
    fm = FillerManager(session, sets, telephony=tel, tts=tts)
    await fm.start()
    # Let some chunks land.
    await asyncio.sleep(0.06)
    await fm.stop()

    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    # Stopped before completion → no event recorded.
    assert fillers == []


async def test_idempotent_start_within_turn() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    sets = [_set(1, (1, "hi", "ready"), (2, "yo", "ready"))]
    fm = FillerManager(session, sets, telephony=tel, tts=tts)
    await fm.start()
    await fm.start()  # second call while task still active is a no-op
    await fm.wait_finished()
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1


pytestmark = pytest.mark.asyncio(loop_scope="session")
