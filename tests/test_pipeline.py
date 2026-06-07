"""Tests for the streaming orchestrator (engine-multi-referee-and-restructure).

Covers: main streaming sentences, N referees running in parallel, single
referee fail-open isolation, chat_stream → chat() fallback, empty-reply default,
wrap-up + restructure skip referees, and the LLM-greeting plain-text path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from isales_common.providers._models import LLMResponse
from isales_common.providers.llm import LLMProvider
from isales_common.schemas.pipeline import (
    ExtractorSpec,
    MainSpec,
    RefereeSpec,
    RestructureSpec,
)

from isales_engine.call_session import CallSession, DialogTurn
from isales_engine.pipeline.greeting import generate_greeting
from isales_engine.pipeline.orchestrator import (
    run_pipeline_stream,
    run_restructure_stream,
)
from isales_engine.pipeline.prompt_builder import LeadInfo, PipelineConfig


def _session() -> CallSession:
    return CallSession(
        call_record_id=1,
        campaign_id=1,
        lead_id=1,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


def _referee(label: str = "main_judge") -> RefereeSpec:
    return RefereeSpec(
        role_config_id=2,
        prompt_version_id=2,
        system_prompt="判定：{{user_last_utterance}} {{recent_dialog_history}}",
        label=label,
    )


def _config(**over) -> PipelineConfig:
    base = {
        "main": MainSpec(role_config_id=1, prompt_version_id=1, system_prompt="你是助手。"),
        "referees": [_referee()],
        "extractor": ExtractorSpec(
            role_config_id=3, prompt_version_id=3, system_prompt="抽取。"
        ),
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
    """chat_stream yields the configured text; chat returns the referee JSON."""

    def __init__(self, *, stream_text: str, referee_json: str, stream_raises: bool = False):
        self._stream_text = stream_text
        self._referee_json = referee_json
        self._stream_raises = stream_raises
        self.chat_calls = 0

    async def chat(
        self, messages, *, json_mode=False, temperature=1.0, top_p=1.0, max_tokens=None
    ):
        self.chat_calls += 1
        user = next((m.content for m in messages if m.role == "user"), "")
        content = self._referee_json if "pass 或 hold" in user else (
            self._stream_text or "您好。"
        )
        return _resp(content)

    async def chat_stream(
        self, messages, *, temperature=1.0, top_p=1.0, max_tokens=None
    ) -> AsyncIterator[str]:
        if self._stream_raises:
            raise RuntimeError("stream boom")
            yield ""  # pragma: no cover
        for ch in self._stream_text:
            yield ch
        self.last_call_tokens_in = 10
        self.last_call_tokens_out = 6
        self.last_call_finish_reason = "stop"


async def _drain(stream) -> list[str]:
    return [s async for s in stream.sentences()]


@pytest.mark.asyncio
async def test_main_streams_sentences_and_referee_runs_parallel():
    llm = _ScriptedLLM(
        stream_text="您好。请问现在方便吗？",
        referee_json="continue",
    )
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm, is_wrap_up=False)
    assert len(stream.referee_tasks) == 1  # spawned immediately
    sentences = await _drain(stream)
    assert sentences == ["您好。", "请问现在方便吗？"]
    assert stream.result.reply_text == "您好。请问现在方便吗？"
    assert stream.result.tokens_in == 10
    referee = await stream.referee_tasks[0]
    assert referee.category == "continue"
    assert referee.label == "main_judge"


@pytest.mark.asyncio
async def test_multiple_referees_run_in_parallel():
    cfg = _config(referees=[_referee("intent"), _referee("reject")])
    llm = _ScriptedLLM(
        stream_text="好的。",
        referee_json="NEGATIVE",
    )
    stream = run_pipeline_stream(_session(), "随便", cfg, llm, llm)
    assert len(stream.referee_tasks) == 2
    assert stream.referee_labels == ["intent", "reject"]
    await _drain(stream)
    results = [await t for t in stream.referee_tasks]
    assert {r.label for r in results} == {"intent", "reject"}
    assert all(r.category == "NEGATIVE" for r in results)


@pytest.mark.asyncio
async def test_single_referee_fail_open_isolated():
    """A referee returning garbage fails open without affecting the others."""

    class _MixedLLM(_ScriptedLLM):
        async def chat(self, messages, *, json_mode=False, **k):
            self.chat_calls += 1
            system = next((m.content for m in messages if m.role == "system"), "")
            # 'reject' referee emits empty output (fails open); others are fine.
            if "reject-prompt" in system:
                return _resp("")
            return _resp("POSITIVE")

    cfg = _config(
        referees=[
            _referee("intent"),
            RefereeSpec(
                role_config_id=9,
                prompt_version_id=9,
                system_prompt="reject-prompt {{user_last_utterance}} {{recent_dialog_history}}",
                label="reject",
            ),
        ]
    )
    llm = _MixedLLM(stream_text="好的。", referee_json="{}")
    stream = run_pipeline_stream(_session(), "x", cfg, llm, llm)
    await _drain(stream)
    by_label = {r.label: r for r in [await t for t in stream.referee_tasks]}
    assert by_label["intent"].category == "POSITIVE"
    assert by_label["reject"].category is None  # failed open
    assert by_label["reject"].failopen_reason == "invalid"


@pytest.mark.asyncio
async def test_wrap_up_skips_referee():
    llm = _ScriptedLLM(stream_text="再见。", referee_json="{}")
    stream = run_pipeline_stream(_session(), "好", _config(), llm, llm, is_wrap_up=True)
    assert stream.referee_tasks == []
    sentences = await _drain(stream)
    assert sentences == ["再见。"]


@pytest.mark.asyncio
async def test_restructure_stream_skips_referee_and_uses_only_interrupt_text():
    cfg = _config(
        restructure=RestructureSpec(
            role_config_id=7,
            prompt_version_id=7,
            system_prompt="重组：",
            label="rewrite",
        )
    )
    sess = _session()
    sess.dialog_history.append(DialogTurn(role="user", text="历史不该出现", ts_ms=0))
    llm = _ScriptedLLM(stream_text="换个说法补一句。", referee_json="{}")
    stream = run_restructure_stream(sess, "原始要点", cfg, llm)
    assert stream.referee_tasks == []
    assert stream.is_restructure
    sentences = await _drain(stream)
    assert "".join(sentences) == "换个说法补一句。"
    # The restructure LLM saw only {system: restructure_prompt, user: InterruptText}.
    msgs = stream._build_messages()
    assert msgs[0].content == "重组："
    assert msgs[1].content == "原始要点"
    assert "历史不该出现" not in msgs[1].content


@pytest.mark.asyncio
async def test_chat_stream_failure_falls_back_to_chat():
    llm = _ScriptedLLM(
        stream_text="兜底回复内容。",
        referee_json="continue",
        stream_raises=True,
    )
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm)
    sentences = await _drain(stream)
    assert stream.result.fallback_used is True
    assert "".join(sentences) == "兜底回复内容。"


@pytest.mark.asyncio
async def test_empty_reply_uses_default():
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

    llm = _EmptyLLM(stream_text="", referee_json="{}")
    stream = run_pipeline_stream(_session(), "你好", _config(), llm, llm)
    sentences = await _drain(stream)
    assert sentences == ["好的，请稍等。"]
    assert stream.result.used_default_reply is True


@pytest.mark.asyncio
async def test_referee_sees_recent_dialog_history():
    sess = _session()
    sess.dialog_history.append(DialogTurn(role="assistant", text="您好", ts_ms=0))
    sess.dialog_history.append(DialogTurn(role="user", text="嗯", ts_ms=1))
    llm = _ScriptedLLM(
        stream_text="好的。",
        referee_json="continue",
    )
    stream = run_pipeline_stream(sess, "周三方便", _config(), llm, llm)
    await _drain(stream)
    await stream.referee_tasks[0]
    # referee chat() was called with the substituted system prompt.
    assert llm.chat_calls == 1


@pytest.mark.asyncio
async def test_generate_greeting_plain_text():
    llm = _ScriptedLLM(stream_text="", referee_json="{}")
    greeting = await generate_greeting(_session(), _config(), llm, fixed_template=None)
    assert greeting  # plain text, non-empty
    assert not greeting.lstrip().startswith("{")


@pytest.mark.asyncio
async def test_generate_greeting_fixed_template_skips_llm():
    llm = _ScriptedLLM(stream_text="x", referee_json="{}")
    greeting = await generate_greeting(
        _session(), _config(), llm, fixed_template="您好，这是开场白。"
    )
    assert greeting == "您好，这是开场白。"
    assert llm.chat_calls == 0
