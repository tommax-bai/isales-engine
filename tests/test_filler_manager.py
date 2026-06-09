"""Tests for FillerManager (filler spec § 垫词池随机不重复)."""

from __future__ import annotations

import asyncio

import pytest

from isales_engine.call_session import CallSession
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.filler_manager import FillerManager
from isales_engine.realtime.mock_telephony import MockTelephonyClient


def _session(call_id: int = 1) -> CallSession:
    return CallSession(
        call_record_id=call_id,
        campaign_id=10,
        lead_id=5,
        caller_id="+x",
        prompt_versions_snapshot={},
    )


async def _new_mock_telephony(call_id: int = 1) -> MockTelephonyClient:
    client = MockTelephonyClient(connect_delay_ms=0)
    await client.dial(call_id, "+x")
    # drain the connected event so the queue is clean
    async for _ in client.events(call_id):
        break
    return client


# ---- selection -------------------------------------------------------------


async def test_filler_skip_when_text_empty() -> None:
    """filler spec § 失败兜底 — empty phrase text is skipped silently."""

    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    fm = FillerManager(session, ["   "], telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()
    assert tel.outbound_log[1] == []
    assert all(e["type"] != "filler" for e in session.full_transcript)


async def test_filler_plays_phrase_and_writes_event() -> None:
    """A non-empty phrase is synthesized live and recorded by text (no id)."""

    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    fm = FillerManager(session, ["嗯稍等"], telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()

    assert len(tel.outbound_log[1]) > 0
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1
    assert fillers[0]["text"] == "嗯稍等"
    assert "filler_phrase_id" not in fillers[0]


async def test_single_pool_no_repeat_until_exhausted() -> None:
    """filler spec § 垫词池随机不重复 — within a call a phrase (by text) isn't
    reused until the whole pool is exhausted, then the used set resets."""

    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=2, chunk_size=4)
    phrases = ["a", "b", "c"]

    chosen: list[str] = []
    for _ in range(7):
        fm = FillerManager(session, phrases, telephony=tel, tts=tts)
        await fm.start()
        await fm.wait_finished()
        last = next(
            e for e in reversed(session.full_transcript) if e["type"] == "filler"
        )
        chosen.append(last["text"])

    # First 3 picks exhaust the pool with no repeats (any order).
    assert sorted(chosen[0:3]) == ["a", "b", "c"]
    # Pool resets → picks 4-6 are again the full pool, no repeats.
    assert sorted(chosen[3:6]) == ["a", "b", "c"]
    # 7th pick starts a fresh cycle.
    assert chosen[6] in {"a", "b", "c"}


async def test_used_set_is_per_call_session() -> None:
    """The 'used' bookkeeping (by text) lives on CallSession, so a fresh call
    starts with the whole pool available again."""

    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=2, chunk_size=4)
    phrases = ["x", "y"]

    s1 = _session(1)
    fm = FillerManager(s1, phrases, telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()
    assert len(s1.used_filler_phrases) == 1

    s2 = _session(1)
    assert s2.used_filler_phrases == set()


async def test_stop_cancels_playback_and_skips_event() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=320, chunk_size=320, chunk_delay_s=0.05)
    fm = FillerManager(session, ["abcdefghijklmnopqrst"], telephony=tel, tts=tts)
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
    fm = FillerManager(session, ["hi", "yo"], telephony=tel, tts=tts)
    await fm.start()
    await fm.start()  # second call while task still active is a no-op
    await fm.wait_finished()
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1


pytestmark = pytest.mark.asyncio(loop_scope="session")
