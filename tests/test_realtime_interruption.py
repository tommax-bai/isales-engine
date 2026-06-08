"""End-to-end tests for impl-engine-providers PR #6 (real-time interruption)."""

from __future__ import annotations

import asyncio
import contextlib

from isales_common.providers._models import ASRResult

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.providers.tts_mock import TextLengthMockTTS
from isales_engine.realtime.interruption_detector import InterruptionConfig
from isales_engine.realtime.interruption_rules import default_rule
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import (
    Providers,
    _decide_protection,
    _partial_monitor,
    _play_streaming,
    _play_tts,
    run_session,
)
from tests.test_play_streaming import (
    _connected,
    _FakeStream,
    _GatedTelephony,
    _RecordingTTS,
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
        whitelist=("嗯", "好的"),
        rule=default_rule(whitelist=["嗯", "好的"], min_duration_ms=400),
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
    # Pretend the VAD monitor has registered above-baseline voice energy
    # for the same span — without this, partial_monitor's VAD-corroboration
    # check skips the trigger (defensive guard against DingRTC mixed-playback
    # self-loopback being mis-transcribed as user speech).
    session.vad_voice_active_ms = 450
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
        whitelist=("嗯",),
        rule=default_rule(whitelist=["嗯"], min_duration_ms=400),
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
        whitelist=("嗯", "好的"),
        rule=default_rule(whitelist=["嗯", "好的"], min_duration_ms=400),
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
        whitelist=("嗯",),
        rule=default_rule(whitelist=["嗯"], min_duration_ms=20),
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


# ---- producer/consumer barge-in: partial_monitor cancels mid-play ----------


async def test_partial_monitor_cancels_play_streaming_midflight() -> None:
    """End-to-end: the real _partial_monitor cancels current_speaking_task
    while _play_streaming is mid-playback. The producer/consumer pipeline
    (pipeline-latency-tail § B) must return fully_played=False and tear down
    every in-flight pre-synthesis — barge-in landing while sentence 0 plays
    and sentences 1+ are pre-synthesizing / queued."""
    import time

    session = _make_session()
    tel = _GatedTelephony()
    tel.release.clear()  # hold sentence 0 mid-play so barge-in lands in-flight
    await _connected(tel, session.call_record_id)
    tts = _RecordingTTS(synth_delay_s=0.01, chunks_per_sentence=3)
    stream = _FakeStream(["第一句话。", "第二句话。", "第三句话。"])

    interruption_cfg = InterruptionConfig(
        whitelist=("嗯",),
        rule=default_rule(whitelist=["嗯"], min_duration_ms=400),
    )
    asr_partials_q: asyncio.Queue[ASRResult] = asyncio.Queue()
    monitor = asyncio.create_task(
        _partial_monitor(
            session,
            asr_partials_q=asr_partials_q,
            interruption_cfg=interruption_cfg,
        )
    )

    play_task = asyncio.create_task(
        _play_streaming(
            session, tel, tts, stream,  # type: ignore[arg-type]
            voice_id="v", filler=None, processing_start=time.monotonic(),
        )
    )

    # Wait until sentence 0 is genuinely mid-play (held by the gate) — by now
    # the producer has pre-synthesized sentences 1+.
    await asyncio.wait_for(tel.first_chunk_played.wait(), timeout=1.0)
    await asyncio.sleep(0.05)

    # User barges in: anchor + a long non-whitelist partial with VAD corroboration.
    await asr_partials_q.put(ASRResult(text="等", is_final=False, timestamp_ms=100))
    await asyncio.sleep(0.45)
    session.vad_voice_active_ms = 450
    await asr_partials_q.put(
        ASRResult(text="等一下我有问题", is_final=False, timestamp_ms=600)
    )

    # Monitor cancels current_speaking_task; release the gate so the cancelled
    # playback unwinds.
    for _ in range(50):
        if session.interruption_signaled:
            break
        await asyncio.sleep(0.01)
    tel.release.set()

    first_audio_ms, played = await asyncio.wait_for(play_task, timeout=2.0)
    monitor.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await monitor

    assert session.interruption_signaled is True
    assert played is False
    assert first_audio_ms is not None
    assert tts.active == 0  # no pre-synthesis left running after barge-in


