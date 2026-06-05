"""Process-level fixed-phrase TTS cache (tts-cache-and-gated-filler § A).

Fixed / reusable phrases — the campaign greeting, silence / transfer /
wrap-up / filler phrases, the "您请说" protection cue — are identical across
every call, yet the engine synthesizes them live each time (~330ms first byte
+ network). ``CachingTTSProvider`` wraps any ``TTSProvider`` and replays the
stored PCM on a cache hit (zero synthesis, zero network); on a miss it passes
through, accumulates the chunks, and fills the cache once synthesis completes.

The cache (``TtsCacheStore``) is a process-level bounded LRU shared across
calls, so the greeting is synthesized once per engine process and free
thereafter. Only short text is cached (``max_cacheable_chars``) so dynamic
main-LLM replies don't pollute or bloat it. Inspired by voxen's
``core/provider/tts/cache``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import AsyncIterator

from isales_common.providers.tts import TTSProvider

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_TOTAL_BYTES = 32 << 20  # 32 MiB
DEFAULT_MAX_CACHEABLE_CHARS = 60


class TtsCacheStore:
    """Process-level bounded LRU cache: (text, voice_id) → list[PCM chunk].

    Bounded by both entry count and total bytes; least-recently-used entries
    are evicted first. Shared across calls (constructed once at process
    startup), so fixed phrases survive between calls.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._store: OrderedDict[tuple[str, str], list[bytes]] = OrderedDict()
        self._total_bytes = 0

    def get(self, key: tuple[str, str]) -> list[bytes] | None:
        chunks = self._store.get(key)
        if chunks is not None:
            self._store.move_to_end(key)  # mark MRU
        return chunks

    def put(self, key: tuple[str, str], chunks: list[bytes]) -> None:
        if key in self._store:
            self._total_bytes -= sum(len(c) for c in self._store[key])
            del self._store[key]
        size = sum(len(c) for c in chunks)
        if size > self._max_total_bytes:
            return  # single entry too big to cache at all
        self._store[key] = chunks
        self._total_bytes += size
        self._evict()

    def _evict(self) -> None:
        while self._store and (
            len(self._store) > self._max_entries
            or self._total_bytes > self._max_total_bytes
        ):
            _key, chunks = self._store.popitem(last=False)  # LRU
            self._total_bytes -= sum(len(c) for c in chunks)

    # Introspection (tests / metrics).
    def __len__(self) -> int:
        return len(self._store)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


class CachingTTSProvider(TTSProvider):
    """Wrap a TTSProvider with a fixed-phrase replay cache."""

    def __init__(
        self,
        inner: TTSProvider,
        store: TtsCacheStore,
        *,
        max_cacheable_chars: int = DEFAULT_MAX_CACHEABLE_CHARS,
    ) -> None:
        self._inner = inner
        self._store = store
        self._max_cacheable_chars = max_cacheable_chars

    async def synthesize_stream(
        self, text: str, voice_id: str
    ) -> AsyncIterator[bytes]:
        key = (text, voice_id)
        hit = self._store.get(key)
        if hit is not None:
            logger.info(
                "tts_cache_hit text_len=%s voice=%s chunks=%s",
                len(text), voice_id, len(hit),
            )
            for chunk in hit:
                yield chunk
            return

        # Don't cache dynamic / long text (main LLM replies) — only short
        # fixed phrases. Pass straight through.
        if len(text) > self._max_cacheable_chars:
            async for chunk in self._inner.synthesize_stream(text, voice_id):
                yield chunk
            return

        # Cacheable miss: accumulate while streaming, store only on a clean
        # full completion (a mid-stream error / barge-in cancel must NOT cache
        # a truncated phrase).
        buf: list[bytes] = []
        async for chunk in self._inner.synthesize_stream(text, voice_id):
            buf.append(chunk)
            yield chunk
        self._store.put(key, buf)
        logger.info(
            "tts_cache_fill text_len=%s voice=%s chunks=%s entries=%s",
            len(text), voice_id, len(buf), len(self._store),
        )

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()
