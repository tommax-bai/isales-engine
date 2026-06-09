"""Tests for FillerManager (filler spec § 垫词池随机不重复)."""

from __future__ import annotations

import asyncio

import pytest

from isales_engine.call_session import CallSession
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.filler_manager import FillerManager, FillerPhraseSpec
from isales_engine.realtime.mock_telephony import MockTelephonyClient


def _session(call_id: int = 1) -> CallSession:
    return CallSession(
        call_record_id=call_id,
        campaign_id=10,
        lead_id=5,
        caller_id="+x",
        prompt_versions_snapshot={},
    )


def _phrases(*phrases: tuple[int, str, str]) -> list[FillerPhraseSpec]:
    """Each tuple is (phrase_id, text, status)."""

    return [
        FillerPhraseSpec(
            id=pid,
            text=text,
            audio_url=f"oss://x/{pid}.wav" if status == "ready" else None,
            generation_status=status,
        )
        for pid, text, status in phrases
    ]


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
    fm = FillerManager(session, _phrases((1, "   ", "pending")), telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()
    assert tel.outbound_log[1] == []
    assert all(e["type"] != "filler" for e in session.full_transcript)


async def test_filler_plays_pending_phrase_via_realtime_synth() -> None:
    """v1.0 (filler spec § 预生成 + 动态补充音频): a phrase with non-empty
    text is synthesized live regardless of audio_url / generation_status —
    gating on those stage-6 OSS fields would make filler never fire."""

    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    # pending → audio_url is None via _phrases helper; must still play.
    fm = FillerManager(session, _phrases((1, "嗯稍等", "pending")), telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()

    assert len(tel.outbound_log[1]) > 0  # PCM forwarded despite no audio_url
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1
    assert fillers[0]["filler_phrase_id"] == 1


async def test_filler_picks_phrase_and_writes_event() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)
    fm = FillerManager(
        session,
        _phrases((1, "嗯", "ready"), (2, "稍等", "ready")),
        telephony=tel,
        tts=tts,
    )
    await fm.start()
    await fm.wait_finished()

    assert len(tel.outbound_log[1]) > 0  # PCM bytes were forwarded
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1
    assert fillers[0]["filler_phrase_id"] in {1, 2}


async def test_single_pool_no_repeat_until_exhausted() -> None:
    """filler spec § 垫词池随机不重复 — within a call a phrase isn't reused
    until the whole per-campaign pool is exhausted, then the used set resets."""

    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=2, chunk_size=4)
    phrases = _phrases((10, "a", "ready"), (11, "b", "ready"), (12, "c", "ready"))

    chosen: list[int] = []
    for _ in range(7):
        fm = FillerManager(session, phrases, telephony=tel, tts=tts)
        await fm.start()
        await fm.wait_finished()
        last = next(
            e for e in reversed(session.full_transcript) if e["type"] == "filler"
        )
        chosen.append(last["filler_phrase_id"])

    # First 3 picks exhaust the pool with no repeats (any order).
    assert sorted(chosen[0:3]) == [10, 11, 12]
    # Pool resets → picks 4-6 are again the full pool, no repeats.
    assert sorted(chosen[3:6]) == [10, 11, 12]
    # 7th pick starts a fresh cycle.
    assert chosen[6] in {10, 11, 12}


async def test_used_set_is_per_call_session() -> None:
    """The 'used' bookkeeping lives on CallSession, so a fresh call starts with
    the whole pool available again (no cross-call leakage)."""

    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=2, chunk_size=4)
    phrases = _phrases((20, "x", "ready"), (21, "y", "ready"))

    s1 = _session(1)
    fm = FillerManager(s1, phrases, telephony=tel, tts=tts)
    await fm.start()
    await fm.wait_finished()
    assert len(s1.used_filler_phrase_ids) == 1

    s2 = _session(1)
    assert s2.used_filler_phrase_ids == set()


async def test_stop_cancels_playback_and_skips_event() -> None:
    session = _session()
    tel = await _new_mock_telephony()
    tts = TextLengthMockTTS(pcm_bytes_per_char=320, chunk_size=320, chunk_delay_s=0.05)
    fm = FillerManager(
        session,
        _phrases((1, "abcdefghijklmnopqrst", "ready")),
        telephony=tel,
        tts=tts,
    )
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
    fm = FillerManager(
        session,
        _phrases((1, "hi", "ready"), (2, "yo", "ready")),
        telephony=tel,
        tts=tts,
    )
    await fm.start()
    await fm.start()  # second call while task still active is a no-op
    await fm.wait_finished()
    fillers = [e for e in session.full_transcript if e["type"] == "filler"]
    assert len(fillers) == 1


pytestmark = pytest.mark.asyncio(loop_scope="session")
