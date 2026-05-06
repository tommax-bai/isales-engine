"""Mock TTS that emits PCM proportional to text length.

Spec: provider-abc § Requirement: TTS Provider 异步流式合成接口.

Each character produces ``pcm_bytes_per_char`` bytes so realtime modules
(SPEAKING / FILLER) can compute deterministic durations during tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from isales_common.providers.tts import TTSProvider


class TextLengthMockTTS(TTSProvider):
    def __init__(
        self,
        *,
        pcm_bytes_per_char: int = 320,  # ~20ms @ 8kHz signed 16-bit
        chunk_size: int = 320,
        chunk_delay_s: float = 0.0,
    ) -> None:
        self._bytes_per_char = pcm_bytes_per_char
        self._chunk_size = chunk_size
        self._chunk_delay_s = chunk_delay_s
        self.calls: list[tuple[str, str]] = []

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
    ) -> AsyncIterator[bytes]:
        self.calls.append((text, voice_id))
        total = max(1, len(text)) * self._bytes_per_char
        emitted = 0
        while emitted < total:
            n = min(self._chunk_size, total - emitted)
            yield bytes(n)
            emitted += n
            if self._chunk_delay_s:
                await asyncio.sleep(self._chunk_delay_s)
