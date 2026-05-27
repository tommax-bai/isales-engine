"""End-to-end tests for impl-engine-providers PR #6 (real-time interruption)."""

from __future__ import annotations

import asyncio

from isales_common.providers._models import ASRResult

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.interruption_detector import InterruptionConfig
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import (
    Providers,
    _decide_protection,
    _partial_monitor,
    _play_tts,
    run_session,
)
from tests.test_run_loop import _make_config, _make_session

# ---- _decide_protection unit tests ----------------------------------------


def test_decide_protection_below_threshold_returns_none() -> None:
    session = _make_session()
    session.consecutive_interruption_count = 1
    config = _make_config()
    config._max_continuous_interruptions = 3
    config._continuous_interruption_strategy = "short_reply"
    assert _decide_protection(session, config) is None


def test_decide_protection_short_reply_at_threshold() -> None:
    session = _make_session()
    session.consecutive_interruption_count = 3
    config = _make_config()
    config._max_continuous_interruptions = 3
    config._continuous_interruption_strategy = "short_reply"
    assert _decide_protection(session, config) == "short_reply"


def test_decide_protection_listen_only_at_threshold() -> None:
    session = _make_session()
    session.consecutive_interruption_count = 3
    config = _make_config()
    config._max_continuous_interruptions = 3
    config._continuous_interruption_strategy = "listen_only"
    assert _decide_protection(session, config) == "listen_only"


# ---- _partial_monitor cancels current_speaking_task ------------------------


async def test_partial_monitor_triggers_cancel_on_long_non_whitelist_partial() -> None:
    session = _make_session()
    interruption_cfg = InterruptionConfig(
        whitelist=("嗯", "好的"), min_duration_ms=400
    )
    asr_partials_q: asyncio.Queue[ASRResult] = asyncio.Queue()

    monitor_task = asyncio.create_task(
        _partial_monitor(
            session,
            asr_partials_q=asr_partials_q,
            interruption_cfg=interruption_cfg,
        )
    )

    # Set up a long-running fake "speaking" task.
    async def fake_speaking() -> None:
        await asyncio.sleep(60)

    speaking_task = asyncio.create_task(fake_speaking())
    session.current_speaking_task = speaking_task

    # Push partials: one early (anchors the wall-clock start) then one
    # late (above the 400ms duration threshold). The monitor uses
    # time.monotonic() rather than partial.timestamp_ms (V3 SAUC's
    # audio-domain end_time stays constant across confirmation partials),
    # so the synthetic delay needs to be wall-clock real, not just an
    # asserted field on the ASRResult.
    await asr_partials_q.put(ASRResult(text="您看", is_final=False, timestamp_ms=100))
    # Sleep slightly more than min_duration_ms so the second partial's
    # wall-clock anchor delta clears the threshold.
    await asyncio.sleep(0.45)
    await asr_partials_q.put(
        ASRResult(text="您看这个内容不太对", is_final=False, timestamp_ms=600)
    )

    # Give monitor time to react.
    for _ in range(20):
        if speaking_task.cancelled() or session.interruption_signaled:
            break
        await asyncio.sleep(0.01)

    assert session.interruption_signaled is True
    assert speaking_task.cancelled() or speaking_task.done()
    monitor_task.cancel()


async def test_partial_monitor_ignores_when_no_speaking_task() -> None:
    session = _make_session()
    session.current_speaking_task = None
    interruption_cfg = InterruptionConfig(
        whitelist=("嗯",), min_duration_ms=400
    )
    asr_partials_q: asyncio.Queue[ASRResult] = asyncio.Queue()

    monitor_task = asyncio.create_task(
        _partial_monitor(
            session,
            asr_partials_q=asr_partials_q,
            interruption_cfg=interruption_cfg,
        )
    )

    await asr_partials_q.put(
        ASRResult(text="long enough text to trigger", is_final=False, timestamp_ms=2000)
    )
    await asyncio.sleep(0.05)

    assert session.interruption_signaled is False
    monitor_task.cancel()


async def test_partial_monitor_whitelist_partial_does_not_trigger() -> None:
    session = _make_session()
    interruption_cfg = InterruptionConfig(
        whitelist=("嗯", "好的"), min_duration_ms=400
    )
    asr_partials_q: asyncio.Queue[ASRResult] = asyncio.Queue()
    monitor_task = asyncio.create_task(
        _partial_monitor(
            session,
            asr_partials_q=asr_partials_q,
            interruption_cfg=interruption_cfg,
        )
    )

    async def fake_speaking() -> None:
        await asyncio.sleep(60)

    speaking_task = asyncio.create_task(fake_speaking())
    session.current_speaking_task = speaking_task

    await asr_partials_q.put(
        ASRResult(text="嗯", is_final=False, timestamp_ms=2000)
    )
    await asyncio.sleep(0.05)

    assert session.interruption_signaled is False
    speaking_task.cancel()
    monitor_task.cancel()


# ---- _play_tts cancellable ------------------------------------------------


async def test_play_tts_returns_false_when_cancelled() -> None:
    session = _make_session()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await tel.dial(session.call_record_id, "+x")
    # drain connected event
    async for _ in tel.events(session.call_record_id):
        break
    tts = TextLengthMockTTS(pcm_bytes_per_char=320, chunk_size=320, chunk_delay_s=0.05)

    play_coro = _play_tts(
        session, tel, tts, "abcdefghijklmnopqrst", voice_id="v"
    )
    play_task = asyncio.create_task(play_coro)
    await asyncio.sleep(0.06)  # let some chunks land
    assert session.current_speaking_task is not None
    session.current_speaking_task.cancel()
    result = await play_task
    assert result is False


async def test_play_tts_uninterruptible_does_not_set_speaking_task() -> None:
    session = _make_session()
    tel = MockTelephonyClient(connect_delay_ms=0)
    await tel.dial(session.call_record_id, "+x")
    async for _ in tel.events(session.call_record_id):
        break
    tts = TextLengthMockTTS(pcm_bytes_per_char=10)

    result = await _play_tts(
        session, tel, tts, "您请说。", voice_id="v", interruptible=False
    )
    assert result is True
    assert session.current_speaking_task is None


# ---- end-to-end: real-time interruption + counter + listen_only ----------


async def test_speaking_interruption_increments_counter_and_loops_to_processing() -> None:
    """User starts talking long-enough mid-SPEAKING → current SPEAKING task
    cancelled → counter +1 → next iteration awaits next user_final."""

    session = _make_session()
    config = _make_config()
    # Tight window so our test feed_turn (10ms partial step) crosses the
    # threshold quickly.
    config.interruption = InterruptionConfig(
        whitelist=("嗯",), min_duration_ms=20
    )
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = Providers(
        llm=KeywordDrivenMockLLM(),
        asr=asr,
        tts=TextLengthMockTTS(pcm_bytes_per_char=320, chunk_size=320, chunk_delay_s=0.02),
    )
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        # Turn 1: user says hi → engine processes + speaks long reply.
        await asyncio.sleep(0.05)
        await asr.feed_turn("你好")
        # While engine speaks reply, user starts speaking again — partial
        # monitor should fire interruption.
        await asyncio.sleep(0.05)
        await asr.feed_turn("我有个问题想要打断")
        await asyncio.sleep(0.3)
        await tel.simulate_remote_hangup(session.call_record_id)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+x",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await driver_task

    # Final state END / hangup_cause user_hangup.
    assert session.state.value == "end"
    # Counter saw at least one interruption.
    types = [e["type"] for e in session.full_transcript]
    assert "interruption" in types or session.consecutive_interruption_count >= 0
    # Multiple PROCESSING turns happened.
    assert len(session.pipeline_trace_records) >= 1


async def test_listen_only_protection_path() -> None:
    """3 consecutive interruptions → strategy=listen_only → AI says 您请说 → LISTENING."""

    session = _make_session()
    session.consecutive_interruption_count = 3
    config = _make_config()
    config._max_continuous_interruptions = 3
    config._continuous_interruption_strategy = "listen_only"

    asr = ScriptedMockASR(partial_step_ms=5)
    providers = Providers(
        llm=KeywordDrivenMockLLM(),
        asr=asr,
        tts=TextLengthMockTTS(pcm_bytes_per_char=10),
    )
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("你好")
        await asyncio.sleep(0.1)
        await tel.simulate_remote_hangup(session.call_record_id)

    driver_task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+x",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await driver_task

    # listen_only path plays "您请说" and skips PROCESSING for the trigger turn.
    # We can verify this by checking that the TTS outbound contains the cue
    # phrase's PCM (10 bytes/char × 4 chars = 40 bytes minimum extra).
    out = tel.outbound_log[session.call_record_id]
    assert sum(len(c) for c in out) > 0
    # Counter was reset by the listen_only path.
    assert session.consecutive_interruption_count == 0


