"""Dual-LLM streaming pipeline orchestrator.

Spec: ai-pipeline § "单 main LLM streaming" / "referee 二级决策"; design.md
决策 2 / 7 / 8.

One ``run_pipeline_stream()`` call drives one PROCESSING turn:

1. Spawn the referee LLM task (side-band enum decision) — unless this is a
   WRAPPING_UP turn, where no state transition is needed.
2. Stream the main LLM reply token-by-token through the sentence splitter; the
   caller (run_loop) feeds each sentence to TTS the moment it is ready.
3. If ``chat_stream`` fails *before any sentence is produced*, fall back once to
   the non-streaming ``chat()`` and emit the whole reply. (This is the single,
   removal-tracked streaming fallback — followup ``pipeline-remove-streaming-
   fallback`` deletes it once the streaming path proves a 30-day SLA.) A failure
   *after* sentences have already played cannot be un-played: we stop and record
   the error.
4. If the reply is still empty (stream + fallback both produced nothing), emit a
   campaign ``default_reply`` so the call never goes silent.

The structured decision (goal_achieved / transfer / customer_decline) comes from
the referee task the caller awaits after playback; the extracted CRM fields are
produced offline by the worker. The main LLM itself emits **plain text only**.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

from isales_common.providers._models import Message
from isales_common.providers.llm import LLMProvider

from isales_engine.call_session import CallSession
from isales_engine.pipeline.prompt_builder import PipelineConfig, build_main_messages
from isales_engine.referee import recent_dialog_rounds, run_referee
from isales_engine.streaming.sentence_splitter import split_sentences
from isales_engine.streaming.types import RefereeResult

logger = logging.getLogger(__name__)


@dataclass
class MainStreamResult:
    """Filled in as ``PipelineStream.sentences()`` is consumed."""

    reply_text: str = ""
    duration_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    fallback_used: bool = False
    used_default_reply: bool = False
    error: str | None = None
    # Timing instrumentation (ms from sentences() start): when the main LLM's
    # first token / first splittable sentence arrived.
    first_token_ms: int | None = None
    first_sentence_ms: int | None = None


class PipelineStream:
    """Handle for one streaming PROCESSING turn.

    ``referee_tasks`` are the side-band decision tasks (empty during wrap-up /
    restructure). ``result`` is populated once ``sentences()`` has been fully
    consumed.
    """

    def __init__(
        self,
        session: CallSession,
        user_input: str,
        config: PipelineConfig,
        main_llm: LLMProvider,
        referee_llm: LLMProvider,
        *,
        is_wrap_up: bool,
        pipeline_timeout_ms: int,
        restructure_text: str | None = None,
    ) -> None:
        self._session = session
        self._user_input = user_input
        self._config = config
        self._main_llm = main_llm
        self._referee_llm = referee_llm
        self._is_wrap_up = is_wrap_up
        self._timeout_s = pipeline_timeout_ms / 1000.0
        # restructure mode (engine-multi-referee-and-restructure D4): when set,
        # this stream re-voices ``restructure_text`` via the restructure slot
        # with no dialog history and no referees.
        self._restructure_text = restructure_text
        # N parallel referee tasks (was a single referee_task). Empty during
        # wrap-up / restructure turns. ``referee_labels`` is the parallel list of
        # labels so the awaiter can build a labelled fail-open on timeout.
        self.referee_tasks: list[asyncio.Task[RefereeResult]] = []
        self.referee_labels: list[str] = []
        self.result = MainStreamResult()
        self.turn_id = 0

    @property
    def is_restructure(self) -> bool:
        return self._restructure_text is not None

    def start(self) -> None:
        """Spawn the referee tasks (concurrent with main streaming).

        One task per configured referee runs in parallel via the event loop
        (awaited together later). No referees during wrap-up or restructure.
        """
        self._session.current_turn_id += 1
        self.turn_id = self._session.current_turn_id
        if self._is_wrap_up or self.is_restructure:
            return
        recent = recent_dialog_rounds(self._session.dialog_history)
        for spec in self._config.referees:
            self.referee_tasks.append(
                asyncio.create_task(
                    run_referee(
                        self._session,
                        self._user_input,
                        recent,
                        spec,
                        self._referee_llm,
                    ),
                    name=f"referee-{spec.label}-turn-{self.turn_id}",
                )
            )
            self.referee_labels.append(spec.label)

    def _stream_params(self) -> tuple[float, float]:
        """(temperature, top_p) for this stream's LLM slot."""
        if self.is_restructure and self._config.restructure is not None:
            return self._config.restructure.temperature, self._config.restructure.top_p
        return self._config.main.temperature, self._config.main.top_p

    def _build_messages(self) -> list[Message]:
        if self.is_restructure and self._config.restructure is not None:
            # D4: restructure input is ONLY {system: restructure_prompt,
            # user: InterruptText} — no dialog_history, no latest utterance.
            return [
                Message(role="system", content=self._config.restructure.system_prompt),
                Message(role="user", content=self._restructure_text or ""),
            ]
        return build_main_messages(
            self._session, self._config, is_wrap_up=self._is_wrap_up
        )

    async def sentences(self) -> AsyncGenerator[str, None]:
        """Yield TTS-ready sentences from the streaming reply (main/restructure)."""
        messages = self._build_messages()
        temperature, top_p = self._stream_params()
        start = time.monotonic()
        yielded_any = False

        async def _timed_tokens() -> AsyncIterator[str]:
            async for tok in self._main_llm.chat_stream(
                messages,
                temperature=temperature,
                top_p=top_p,
            ):
                if self.result.first_token_ms is None:
                    self.result.first_token_ms = int(
                        (time.monotonic() - start) * 1000
                    )
                yield tok

        try:
            async for sentence in split_sentences(_timed_tokens()):
                if self.result.first_sentence_ms is None:
                    self.result.first_sentence_ms = int(
                        (time.monotonic() - start) * 1000
                    )
                yielded_any = True
                self.result.reply_text += sentence
                yield sentence
        except Exception as exc:  # noqa: BLE001 — must not crash the turn
            if yielded_any:
                # Some audio already played; can't un-play. Record + stop.
                logger.warning(
                    "main_stream_failed_mid_turn turn=%s: %s", self.turn_id, exc
                )
                self.result.error = f"main_stream_mid: {exc}"
            else:
                logger.warning(
                    "main_stream_failed_before_first_token turn=%s; chat() fallback",
                    self.turn_id,
                )
                async for sentence in self._chat_fallback(messages):
                    yielded_any = True
                    self.result.reply_text += sentence
                    yield sentence
        finally:
            self.result.duration_ms = int((time.monotonic() - start) * 1000)
            self.result.tokens_in = self._main_llm.last_call_tokens_in or 0
            self.result.tokens_out = self._main_llm.last_call_tokens_out or 0

        if not self.result.reply_text.strip():
            # Stream + fallback both silent → default reply, never go silent.
            default = (
                random.choice(self._config.default_replies)
                if self._config.default_replies
                else "好的，请稍等。"
            )
            self.result.reply_text = default
            self.result.used_default_reply = True
            if self.result.error is None:
                self.result.error = "empty_reply_default_used"
            self._session.append_event(
                "default_reply_used", text=default, reason=self.result.error
            )
            yield default

    async def _chat_fallback(self, messages: list[Message]) -> AsyncIterator[str]:
        """One-shot non-streaming fallback (removal-tracked)."""
        temperature, top_p = self._stream_params()
        try:
            async with asyncio.timeout(self._timeout_s):
                resp = await self._main_llm.chat(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("main_chat_fallback_failed turn=%s: %s", self.turn_id, exc)
            self.result.error = f"main_fallback: {exc}"
            return
        self.result.fallback_used = True
        text = (resp.content or "").strip()
        if text:
            # Re-split so long fallback replies still chunk for TTS.
            async def _one() -> AsyncIterator[str]:
                yield text

            async for sentence in split_sentences(_one()):
                yield sentence


def run_pipeline_stream(
    session: CallSession,
    user_input: str,
    config: PipelineConfig,
    main_llm: LLMProvider,
    referee_llm: LLMProvider,
    *,
    is_wrap_up: bool = False,
    pipeline_timeout_ms: int = 8000,
) -> PipelineStream:
    """Build + start a :class:`PipelineStream` (referee spawned immediately)."""
    stream = PipelineStream(
        session,
        user_input,
        config,
        main_llm,
        referee_llm,
        is_wrap_up=is_wrap_up,
        pipeline_timeout_ms=pipeline_timeout_ms,
    )
    stream.start()
    return stream


def run_restructure_stream(
    session: CallSession,
    restructure_text: str,
    config: PipelineConfig,
    llm: LLMProvider,
    *,
    pipeline_timeout_ms: int = 8000,
) -> PipelineStream:
    """Build + start a restructure :class:`PipelineStream` (no referees).

    Re-voices ``restructure_text`` via the restructure slot. The caller plays
    it through the same ``_play_streaming`` path and returns to LISTENING
    without running referees (D4).
    """
    stream = PipelineStream(
        session,
        "",  # restructure ignores user_input
        config,
        llm,
        llm,
        is_wrap_up=False,
        pipeline_timeout_ms=pipeline_timeout_ms,
        restructure_text=restructure_text,
    )
    stream.start()
    return stream


__all__ = [
    "PipelineStream",
    "MainStreamResult",
    "run_pipeline_stream",
    "run_restructure_stream",
    "PipelineConfig",
]
