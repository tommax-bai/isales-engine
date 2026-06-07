"""Eager speculative buffering for the pre-reply referee gate.

engine-tools-multidialogue-gating §6.1 / §6.7 / §7.2. The dialogue route
generates its reply EAGERLY into a replay buffer the moment the user final lands
(overlapping the referee gate, ~0ms p50), but the generator handed to playback
is the fresh ``sentences()`` replay — un-iterated (AGEN_CREATED). This guards the
blueprint's #1 risk: a "tidy-up" that drains the live generator in the router
would silence every turn. The eager generation runs on the hidden
``_generate_core`` task; the handed-off generator is never pre-iterated.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest
from isales_common.providers._models import LLMResponse
from isales_common.providers.llm import LLMProvider
from isales_common.schemas.pipeline import ExtractorSpec, MainSpec, RefereeSpec

from isales_engine.call_session import CallSession
from isales_engine.pipeline.orchestrator import run_pipeline_stream
from isales_engine.pipeline.prompt_builder import LeadInfo, PipelineConfig


def _session() -> CallSession:
    return CallSession(
        call_record_id=1,
        campaign_id=1,
        lead_id=1,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


def _config(**over) -> PipelineConfig:
    base = {
        "main": MainSpec(role_config_id=1, prompt_version_id=1, system_prompt="你是助手。"),
        "referees": [
            RefereeSpec(
                role_config_id=2,
                prompt_version_id=2,
                system_prompt="判定：{{user_last_utterance}} {{recent_dialog_history}}",
                label="main_judge",
            )
        ],
        "extractor": ExtractorSpec(role_config_id=3, prompt_version_id=3, system_prompt="抽取。"),
        "default_replies": ["好的，请稍等。"],
        "lead": LeadInfo(name="李四", phone="+86138", custom_data={}),
    }
    base.update(over)
    return PipelineConfig(**base)


def _resp(content: str) -> LLMResponse:
    return LLMResponse(
        content=content, tokens_in=5, tokens_out=3, finish_reason="stop", latency_ms=0
    )


class _ScriptedLLM(LLMProvider):
    def __init__(self, *, stream_text: str, referee_json: str = "{}", char_delay_s: float = 0.0):
        self._stream_text = stream_text
        self._referee_json = referee_json
        self._char_delay_s = char_delay_s
        self.chat_calls = 0

    async def chat(
        self, messages, *, json_mode=False, temperature=1.0, top_p=1.0, max_tokens=None
    ):
        self.chat_calls += 1
        user = next((m.content for m in messages if m.role == "user"), "")
        content = (
            self._referee_json
            if "JSON schema 输出" in user
            else (self._stream_text or "您好。")
        )
        return _resp(content)

    async def chat_stream(
        self, messages, *, temperature=1.0, top_p=1.0, max_tokens=None
    ) -> AsyncIterator[str]:
        for ch in self._stream_text:
            if self._char_delay_s:
                await asyncio.sleep(self._char_delay_s)
            yield ch
        self.last_call_tokens_in = 10
        self.last_call_tokens_out = 6
        self.last_call_finish_reason = "stop"


@pytest.mark.asyncio
async def test_eager_dialogue_route_returns_live_generator():
    """The #1 footgun guard: the generator handed to playback is AGEN_CREATED.

    Even though ``start_eager()`` is already driving generation on the hidden
    task, ``sentences()`` returns a fresh, un-iterated replay generator. A future
    refactor that drains/awaits it in the router would flip this to
    SUSPENDED/CLOSED and silence every turn.
    """
    llm = _ScriptedLLM(stream_text="您好。请问现在方便吗？")
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm)
    stream.start_eager()

    gen = stream.sentences()
    try:
        assert inspect.getasyncgenstate(gen) == inspect.AGEN_CREATED
    finally:
        await gen.aclose()
        await stream.cancel_eager()


@pytest.mark.asyncio
async def test_eager_replay_yields_same_sentences_as_direct():
    """Replaying the eager buffer is byte-identical to direct generation."""
    llm = _ScriptedLLM(stream_text="您好。请问现在方便吗？")
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm)
    stream.start_eager()

    sentences = [s async for s in stream.sentences()]
    assert sentences == ["您好。", "请问现在方便吗？"]
    assert stream.result.reply_text == "您好。请问现在方便吗？"
    assert stream.result.tokens_in == 10
    await stream.cancel_eager()


@pytest.mark.asyncio
async def test_eager_generation_overlaps_the_gate():
    """Generation runs to completion WITHOUT anyone consuming sentences() — the
    whole point of eager buffering (it overlaps the referee gate)."""
    llm = _ScriptedLLM(stream_text="第一句。第二句。")
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm)
    stream.start_eager()

    # Nobody called sentences() yet; the eager task generates anyway.
    assert stream._eager_task is not None
    await stream._eager_task
    # Nothing replayed → the full reply is still pending playback.
    assert stream.buffer_remainder() == "第一句。第二句。"
    assert stream.result.reply_text == "第一句。第二句。"


@pytest.mark.asyncio
async def test_eager_loser_cancelled_stops_generation():
    """A non-selected speculative route is cancelled mid-flight (loser)."""
    llm = _ScriptedLLM(stream_text="一二三四五六七八九十。", char_delay_s=0.02)
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm)
    stream.start_eager()
    await asyncio.sleep(0.03)  # let a little generation happen

    await stream.cancel_eager()
    assert stream._eager_task is not None
    assert stream._eager_task.done()
    # Cancelled before the full reply was generated.
    assert "十。" not in stream.result.reply_text


@pytest.mark.asyncio
async def test_eager_default_reply_only_fires_for_the_winner():
    """A cancelled loser MUST NOT emit default_reply_used; only the winner that
    actually calls sentences() does."""

    class _EmptyLLM(_ScriptedLLM):
        async def chat_stream(
            self, messages, *, temperature=1.0, top_p=1.0, max_tokens=None
        ):
            self.last_call_tokens_in = 0
            self.last_call_tokens_out = 0
            return
            yield ""  # pragma: no cover

        async def chat(
            self, messages, *, json_mode=False, temperature=1.0, top_p=1.0, max_tokens=None
        ):
            return _resp("")

    # Loser: eager-buffered then cancelled, never played → no default event.
    loser_sess = _session()
    loser = run_pipeline_stream(
        loser_sess, "你好", _config(), _EmptyLLM(stream_text=""), _EmptyLLM(stream_text="")
    )
    loser.start_eager()
    await loser.cancel_eager()
    assert not any(e["type"] == "default_reply_used" for e in loser_sess.full_transcript)

    # Winner: eager-buffered then played → default_reply_used fires once.
    win_sess = _session()
    winner = run_pipeline_stream(
        win_sess, "你好", _config(), _EmptyLLM(stream_text=""), _EmptyLLM(stream_text="")
    )
    winner.start_eager()
    sentences = [s async for s in winner.sentences()]
    assert sentences == ["好的，请稍等。"]
    assert winner.result.used_default_reply is True
    assert sum(e["type"] == "default_reply_used" for e in win_sess.full_transcript) == 1
    await winner.cancel_eager()
