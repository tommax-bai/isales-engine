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
import contextlib
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
        # ---- eager speculative buffering (engine-tools-multidialogue-gating) ----
        # When ``start_eager()`` is called the reply is generated NOW into a
        # replay buffer, concurrent with the pre-reply referee gate, so the gate
        # adds ~0ms p50 (overlap). On a multi-route fan-out every candidate
        # buffers in parallel; the gate selects one and ``cancel_eager()`` cancels
        # the losers (the vendor bills the cancelled tokens — opt-in, cap ≤3).
        # The SELECTED route's ``sentences()`` REPLAYS this buffer and is handed
        # to ``_play_streaming`` un-iterated (AGEN_CREATED) — the eager generation
        # runs on the hidden ``_generate_core()`` task, NEVER on the handed-off
        # generator (the blueprint's #1 "eager-generator" footgun guard).
        self._buffer: list[str] = []
        self._buffer_cond: asyncio.Condition | None = None
        self._eager_task: asyncio.Task[None] | None = None
        self._eager_done: bool = False
        self._replayed_count: int = 0

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

    async def _generate_core(self) -> AsyncGenerator[str, None]:
        """Pure token→sentence generation: NO default-reply, NO transcript events.

        Builds ``result.reply_text`` / tokens / timing and runs the single,
        removal-tracked ``chat_stream → chat()`` fallback. Shared by both the
        direct ``sentences()`` path (legacy / flag-OFF, byte-identical) and the
        eager buffer (``start_eager``). It is side-effect-free w.r.t. the session
        so a *speculative loser* route can run it and be cancelled without ever
        emitting a ``default_reply_used`` event — that winner-only side effect
        lives in ``sentences()`` below.
        """
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

    async def sentences(self) -> AsyncGenerator[str, None]:
        """Yield TTS-ready sentences for the SELECTED route, handed to playback.

        In eager mode (``start_eager`` was called) this REPLAYS the buffer the
        background ``_generate_core`` task is filling; otherwise it drives
        ``_generate_core`` directly (legacy / flag-OFF, byte-identical). Either
        way the generator object returned here is fresh/un-iterated
        (AGEN_CREATED) at hand-off — the eager generation never touches THIS
        generator (blueprint risk #1: a "fix" that drains it silences the turn).
        The default-reply guard + ``default_reply_used`` event fire here, so only
        the winner (which alone calls ``sentences()``) ever emits them.
        """
        if self._eager_task is not None:
            async for sentence in self._replay():
                yield sentence
        else:
            async for sentence in self._generate_core():
                yield sentence

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

    @property
    def is_eager(self) -> bool:
        """True once ``start_eager()`` kicked off speculative buffering."""
        return self._eager_task is not None

    def start_eager(self) -> None:
        """Begin generating the reply NOW into a replay buffer (overlaps the gate).

        Idempotent-guarded by the caller (run_loop spawns one per candidate
        route). Must run inside the event loop (creates the buffer condition +
        task). Restructure / wrap-up streams have no referees but MAY still be
        eager-buffered; the gate simply selects them directly.
        """
        if self._eager_task is not None:
            return
        self._buffer_cond = asyncio.Condition()

        async def _drain_eager() -> None:
            assert self._buffer_cond is not None
            try:
                async for sentence in self._generate_core():
                    async with self._buffer_cond:
                        self._buffer.append(sentence)
                        self._buffer_cond.notify_all()
            finally:
                async with self._buffer_cond:
                    self._eager_done = True
                    self._buffer_cond.notify_all()

        self._eager_task = asyncio.create_task(
            _drain_eager(), name=f"eager-turn-{self.turn_id}"
        )

    async def _replay(self) -> AsyncGenerator[str, None]:
        """Replay the eager buffer + live tail. Created fresh per hand-off
        (AGEN_CREATED) so the playback owner drives it byte-identically to the
        direct path; the underlying generation runs on the ``_drain_eager`` task.
        """
        assert self._buffer_cond is not None
        while True:
            async with self._buffer_cond:
                while (
                    self._replayed_count >= len(self._buffer)
                    and not self._eager_done
                ):
                    await self._buffer_cond.wait()
                if self._replayed_count >= len(self._buffer):
                    return  # eager done, nothing more buffered
                sentence = self._buffer[self._replayed_count]
                self._replayed_count += 1
            yield sentence

    def buffer_remainder(self) -> str:
        """Eager sentences generated but not yet replayed to playback — used by
        ``_play_streaming`` to extend the barge-in ``interrupt_remaining`` capture
        beyond the bounded job queue (the eager buffer can lead it). Empty for the
        non-eager path."""
        return "".join(self._buffer[self._replayed_count :])

    async def cancel_eager(self) -> None:
        """Cancel the speculative generation (loser route, or winner teardown on
        barge-in) so its ``chat_stream`` + vendor TTS connections are released.
        No-op for the non-eager path."""
        if self._eager_task is not None and not self._eager_task.done():
            self._eager_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._eager_task

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
