"""Scripted mock ASR for engine tests.

Spec: provider-abc § Requirement: ASR Provider 异步流式接口;
      interruption-detection § Requirement: 双条件打断判定 (mock supplies the
      partial/final stream the detector consumes).

The default mock from isales-common consumes audio and replays a fixed list
of results once. ``ScriptedMockASR`` keeps the same shape but supports
multiple "user turns" injected mid-call: every call to ``feed_turn(text)``
appends partial + final results to an internal queue, so the orchestrator
sees a realistic LISTENING → speech_end → PROCESSING flow.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from isales_common.providers._models import ASRResult
from isales_common.providers.asr import ASRProvider


class ScriptedMockASR(ASRProvider):
    """Replays results from a per-instance queue.

    Tests drive this via ``feed_turn``; production-style usage is exercised
    by the end-to-end harness in PR #11.
    """

    def __init__(self, *, partial_step_ms: int = 200) -> None:
        self._queue: asyncio.Queue[ASRResult] = asyncio.Queue()
        self._closed = False
        self._partial_step_ms = partial_step_ms
        self.received_chunks: list[bytes] = []

    async def feed_turn(
        self, text: str, *, start_ms: int = 0, final_at_ms: int | None = None
    ) -> None:
        """Push partial results every ``partial_step_ms`` and one final result."""

        chars = list(text)
        for i in range(1, len(chars) + 1):
            await self._queue.put(
                ASRResult(
                    text="".join(chars[:i]),
                    is_final=False,
                    timestamp_ms=start_ms + i * self._partial_step_ms,
                )
            )
        await self._queue.put(
            ASRResult(
                text=text,
                is_final=True,
                timestamp_ms=final_at_ms
                or (start_ms + (len(chars) + 1) * self._partial_step_ms),
            )
        )

    async def close(self) -> None:
        self._closed = True
        # Sentinel so any pending consumer can exit.
        await self._queue.put(
            ASRResult(text="", is_final=True, timestamp_ms=0)
        )

    async def stream_recognize(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[ASRResult]:
        async def _drain() -> None:
            async for chunk in audio_chunks:
                self.received_chunks.append(chunk)

        drain_task = asyncio.create_task(_drain())
        try:
            while True:
                if self._closed and self._queue.empty():
                    return
                try:
                    async with asyncio.timeout(0.05):
                        result = await self._queue.get()
                except TimeoutError:
                    if self._closed:
                        return
                    continue
                if self._closed and result.text == "" and result.is_final:
                    return
                yield result
        finally:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
