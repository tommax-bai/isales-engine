"""Filler manager — overlaps audio with the AI pipeline.

Spec: filler § all Requirements.

The orchestrator (PR #6) launches a PROCESSING turn; in parallel the
``FillerManager`` selects a phrase, synthesizes its audio in real time via
the (cache-wrapped) TTS provider, and streams it via the telephony client.
When the pipeline returns a reply, ``CallSession.run()`` (PR #11) calls
``await wait_finished()`` before starting the reply TTS so the conversation
rhythm stays consistent.

Selection rules (filler spec § 多 filler_set 轮询 + § 集合内随机不重复):

* Walk ``filler_sets`` ordered by ``sort_order`` (id tie-breaker for
  stability), wrapping around forever.
* Each call session keeps a per-set "used" set of phrase ids; pick one not
  in that set; if all are used, reset and pick again.
* Phrases with empty text are skipped silently — no anonymous fallback.
  ``audio_url`` / ``generation_status`` are *not* selection gates in v1.0:
  they belong to the stage-6 OSS pre-render path, which has no producer yet,
  so gating on them would make filler never fire (see filler spec §
  预生成 + 动态补充音频).

The manager is *passive* w.r.t. interruption: when the interruption detector
fires (PR #8) it calls ``stop()`` which cancels the playback task.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

from isales_common.providers.tts import TTSProvider

from isales_engine.call_session import CallSession
from isales_engine.realtime.telephony_client import TelephonyClient

logger = logging.getLogger(__name__)


@dataclass
class FillerPhraseSpec:
    id: int
    text: str
    audio_url: str | None
    generation_status: str  # "pending" / "ready" / "failed"


@dataclass
class FillerSetSpec:
    id: int
    sort_order: int
    phrases: list[FillerPhraseSpec]


# ---- audio loader contract -------------------------------------------------


AudioLoader = "callable taking phrase_id+url, returning bytes; None on failure"


class FillerManager:
    """Owns one in-flight filler audio task per session."""

    def __init__(
        self,
        session: CallSession,
        sets: list[FillerSetSpec],
        *,
        telephony: TelephonyClient,
        tts: TTSProvider,
        voice_id: str = "default",
    ) -> None:
        self._session = session
        self._sets = self._sorted_sets(sets)
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

    def _pick_phrase(self) -> FillerPhraseSpec | None:
        if not self._sets:
            return None

        for _ in range(len(self._sets)):
            set_idx = self._session.current_filler_set_index % len(self._sets)
            chosen_set = self._sets[set_idx]
            self._session.current_filler_set_index += 1

            # v1.0 selects on text only — the phrase is synthesized live in
            # _stream_audio. audio_url / generation_status are stage-6 OSS
            # fields with no producer yet; gating on them = filler never fires.
            # Stage-6 (OSS + regenerate_filler_audio worker) restores an
            # "audio_url ready → stream, else synth" branch in a separate change.
            ready = [p for p in chosen_set.phrases if p.text.strip()]
            if not ready:
                continue

            used = self._session.used_filler_phrase_ids_per_set.setdefault(
                chosen_set.id, set()
            )
            unused = [p for p in ready if p.id not in used]
            if not unused:
                used.clear()
                unused = ready

            phrase = random.choice(unused)
            used.add(phrase.id)
            return phrase

        return None

    # ---- playback --------------------------------------------------------

    async def _play(self, phrase: FillerPhraseSpec) -> None:
        try:
            duration_ms = await self._stream_audio(phrase)
        except asyncio.CancelledError:
            logger.info(
                "filler_cancelled call_record_id=%s phrase_id=%s",
                self._session.call_record_id,
                phrase.id,
            )
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "filler_play_failed call_record_id=%s phrase_id=%s",
                self._session.call_record_id,
                phrase.id,
            )
            return

        # Only record the event if the playback ran to completion (not
        # interrupted) — filler spec § Scenario "垫词期间被打断" implies
        # the cancelled phrase doesn't get logged as having played.
        if not self._stopped:
            self._session.append_event(
                "filler",
                text=phrase.text,
                filler_phrase_id=phrase.id,
                duration_ms=duration_ms,
            )

    async def _stream_audio(self, phrase: FillerPhraseSpec) -> int:
        """Synthesize the phrase live via TTS and forward chunks to telephony.

        v1.0 path: the TTS provider is cache-wrapped (CachingTTSProvider), so
        a repeated (text, voice_id) replays cached PCM with no re-synthesis.
        ``phrase.audio_url`` is intentionally unused here — stage-6 (OSS +
        regenerate_filler_audio worker) will add an "audio_url ready → stream
        the pre-rendered file, else synth" branch in a separate change once
        that worker exists.
        """

        async def chunks() -> AsyncIterator[bytes]:
            async for chunk in self._tts.synthesize_stream(phrase.text, self._voice_id):
                yield chunk

        # Approximate duration: each char in TextLengthMockTTS is ~20ms.
        approx_duration_ms = max(50, len(phrase.text) * 20)
        await self._telephony.audio_out(self._session.call_record_id, chunks())
        return approx_duration_ms

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _sorted_sets(sets: list[FillerSetSpec]) -> list[FillerSetSpec]:
        return sorted(sets, key=lambda s: (s.sort_order, s.id))
