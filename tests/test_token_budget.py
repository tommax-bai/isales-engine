"""Token usage accumulation + budget WARN tests (impl-engine-providers PR #7)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider

from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import Providers, run_session
from tests.test_run_loop import _make_config, _make_session


class _CountingLLM(LLMProvider):
    """Reports tokens_in/out so run_loop accumulates them into
    ``session.total_tokens_in/out``. The main streaming path is the one the
    engine bills against (referee tokens are not accumulated)."""

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
        user = next((m.content for m in messages if m.role == "user"), "")
        if "JSON schema 输出决策" in user:
            content = '{"decision": "continue", "goal_type": null, "confidence": 0.9}'
        else:
            content = "您好。"
        return LLMResponse(
            content=content,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            finish_reason="stop",
            latency_ms=10,
        )

    async def chat_stream(  # type: ignore[override]
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        for ch in "好的。":
            yield ch
        self.last_call_tokens_in = self.tokens_in
        self.last_call_tokens_out = self.tokens_out
        self.last_call_finish_reason = "stop"


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
