"""Filler manager — overlaps audio with the AI pipeline.

Spec: filler § all Requirements.

The orchestrator (PR #6) launches a PROCESSING turn; in parallel the
``FillerManager`` selects a phrase, synthesizes its audio in real time via
the (cache-wrapped) TTS provider, and streams it via the telephony client.
When the pipeline returns a reply, ``CallSession.run()`` (PR #11) calls
``await wait_finished()`` before starting the reply TTS so the conversation
rhythm stays consistent.

Selection rules (filler spec § 垫词池随机不重复):

* A campaign owns a single flat pool of filler phrases — plain strings from the
  ``campaign.filler_phrases`` JSONB column (filler-campaign-column; no separate
  table, no per-phrase audio/id).
* Each call session keeps one ``used`` set of phrase *text*; pick a phrase not
  in that set; if all are used, reset and pick again.
* Phrases with empty text are skipped silently — no anonymous fallback. v1.0
  always synthesizes the chosen text live (filler spec § 运行时合成垫词音频);
  there are no pre-render fields to gate on.

The manager is *passive* w.r.t. interruption: when the interruption detector
fires (PR #8) it calls ``stop()`` which cancels the playback task.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator

from isales_common.providers.tts import TTSProvider

from isales_engine.call_session import CallSession
from isales_engine.realtime.telephony_client import TelephonyClient

logger = logging.getLogger(__name__)


class FillerManager:
    """Owns one in-flight filler audio task per session."""

    def __init__(
        self,
        session: CallSession,
        phrases: list[str],
        *,
        telephony: TelephonyClient,
        tts: TTSProvider,
        voice_id: str = "default",
    ) -> None:
        self._session = session
        self._phrases = phrases
        self._telephony = telephony
        self._tts = tts
        self._voice_id = voice_id
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    # ---- public ----------------------------------------------------------

    async def start(self) -> None:
        """Pick a phrase + spawn audio playback. Idempotent within a turn."""

        if self._task is not None and not self._task.done():
            return

        phrase = self._pick_phrase()
        if phrase is None:
            logger.info(
                "filler_skip_no_ready_phrase call_record_id=%s",
                self._session.call_record_id,
            )
            return

        self._stopped = False
        self._task = asyncio.create_task(self._play(phrase), name="filler_play")

    async def stop(self) -> None:
        """Cancel any in-flight playback (interruption / pipeline-cancelled)."""

        self._stopped = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def wait_finished(self) -> None:
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    # ---- selection -------------------------------------------------------

    def _pick_phrase(self) -> str | None:
        # Single flat per-call pool: don't repeat a phrase (by text) until the
        # pool is exhausted, then reset (filler spec § 垫词池随机不重复). The
        # text is synthesized live in _stream_audio — there are no pre-render
        # fields to gate on (filler spec § 运行时合成垫词音频).
        ready = [p for p in self._phrases if p.strip()]
        if not ready:
            return None

        used = self._session.used_filler_phrases
        unused = [p for p in ready if p not in used]
        if not unused:
            used.clear()
            unused = ready

        phrase = random.choice(unused)
        used.add(phrase)
        return phrase

    # ---- playback --------------------------------------------------------

    async def _play(self, phrase: str) -> None:
        try:
            duration_ms = await self._stream_audio(phrase)
        except asyncio.CancelledError:
            logger.info(
                "filler_cancelled call_record_id=%s text=%r",
                self._session.call_record_id,
                phrase,
            )
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "filler_play_failed call_record_id=%s text=%r",
                self._session.call_record_id,
                phrase,
            )
            return

        # Only record the event if the playback ran to completion (not
        # interrupted) — filler spec § Scenario "垫词期间被打断" implies
        # the cancelled phrase doesn't get logged as having played.
        if not self._stopped:
            self._session.append_event(
                "filler",
                text=phrase,
                duration_ms=duration_ms,
            )

    async def _stream_audio(self, phrase: str) -> int:
        """Synthesize the phrase live via TTS and forward chunks to telephony.

        The TTS provider is cache-wrapped (CachingTTSProvider), so a repeated
        (text, voice_id) replays cached PCM with no re-synthesis (filler spec §
        运行时合成垫词音频).
        """

        async def chunks() -> AsyncIterator[bytes]:
            async for chunk in self._tts.synthesize_stream(phrase, self._voice_id):
                yield chunk

        # Approximate duration: each char in TextLengthMockTTS is ~20ms.
        approx_duration_ms = max(50, len(phrase) * 20)
        await self._telephony.audio_out(self._session.call_record_id, chunks())
        return approx_duration_ms
