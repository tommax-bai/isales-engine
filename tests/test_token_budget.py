"""Token usage accumulation + budget WARN tests (impl-engine-providers PR #7)."""

from __future__ import annotations

import asyncio
import logging

import pytest
from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider

from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import Providers, run_session
from tests.test_run_loop import _make_config, _make_session


class _CountingLLM(LLMProvider):
    """Echoes tokens_in/out from a fixed response so the orchestrator
    accumulates them into ``session.total_tokens_in/out``."""

    def __init__(self, *, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    async def chat(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system = next((m.content for m in messages if m.role == "system"), "")
        if "[judge]" in system:
            content = '{"passed": true, "reason": "ok"}'
        elif "[polish]" in system:
            content = '{"reply": "ok", "selected_candidate_index": 0}'
        else:
            content = (
                '{"reply": "ok", "goal_achieved": false, '
                '"goal_type": "", "extracted": {}}'
            )
        return LLMResponse(
            content=content,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            finish_reason="stop",
            latency_ms=10,
        )


async def test_total_tokens_accumulate_across_pipeline() -> None:
    from isales_engine.providers.asr_mock import ScriptedMockASR
    from isales_engine.providers.tts_mock import TextLengthMockTTS

    session = _make_session()
    config = _make_config()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = Providers(
        llm=_CountingLLM(tokens_in=120, tokens_out=80),
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
        token_budget_per_call=10_000,
    )
    await driver_task

    # 1 turn × N=1 role candidate (config defaults). At least 120/80 each.
    assert session.total_tokens_in >= 120
    assert session.total_tokens_out >= 80


async def test_token_budget_warn_logged_on_overage(caplog) -> None:  # type: ignore[no-untyped-def]
    from isales_engine.providers.asr_mock import ScriptedMockASR
    from isales_engine.providers.tts_mock import TextLengthMockTTS

    session = _make_session()
    config = _make_config()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = Providers(
        llm=_CountingLLM(tokens_in=2000, tokens_out=2000),
        asr=asr,
        tts=TextLengthMockTTS(pcm_bytes_per_char=10),
    )
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("你好")
        await asyncio.sleep(0.1)
        await tel.simulate_remote_hangup(session.call_record_id)

    with caplog.at_level(logging.WARNING, logger="isales_engine.run_loop"):
        driver_task = asyncio.create_task(driver())
        await run_session(
            session,
            phone="+x",
            config=config,
            telephony=tel,
            providers=providers,
            token_budget_per_call=100,  # tiny so any traffic blows it
        )
        await driver_task

    assert any(
        "token_budget_exceeded" in record.message for record in caplog.records
    )


async def test_token_budget_below_threshold_no_warn(caplog) -> None:  # type: ignore[no-untyped-def]
    from isales_engine.providers.asr_mock import ScriptedMockASR
    from isales_engine.providers.tts_mock import TextLengthMockTTS

    session = _make_session()
    config = _make_config()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = Providers(
        llm=_CountingLLM(tokens_in=10, tokens_out=10),
        asr=asr,
        tts=TextLengthMockTTS(pcm_bytes_per_char=10),
    )
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("你好")
        await asyncio.sleep(0.05)
        await tel.simulate_remote_hangup(session.call_record_id)

    with caplog.at_level(logging.WARNING, logger="isales_engine.run_loop"):
        driver_task = asyncio.create_task(driver())
        await run_session(
            session,
            phone="+x",
            config=config,
            telephony=tel,
            providers=providers,
            token_budget_per_call=10_000,
        )
        await driver_task

    assert not any(
        "token_budget_exceeded" in record.message for record in caplog.records
    )


pytestmark = [pytest.mark.asyncio(loop_scope="session")]
