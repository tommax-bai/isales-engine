"""Tests for the fixed-phrase TTS cache (tts-cache-and-gated-filler § A)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from isales_common.providers.tts import TTSProvider

from isales_engine.providers.tts_cache import CachingTTSProvider, TtsCacheStore


class _SpyTTS(TTSProvider):
    """Counts synthesize_stream calls; emits deterministic PCM per text."""

    def __init__(self, *, chunks: int = 3) -> None:
        self.calls: list[tuple[str, str]] = []
        self._chunks = chunks
        self.closed = False
        self.fail_on: set[str] = set()

    async def synthesize_stream(
        self, text: str, voice_id: str
    ) -> AsyncIterator[bytes]:
        self.calls.append((text, voice_id))
        if text in self.fail_on:
            raise RuntimeError(f"boom: {text}")
        for i in range(self._chunks):
            yield f"{text}:{voice_id}:{i}".encode()

    async def aclose(self) -> None:
        self.closed = True


async def _drain(provider: CachingTTSProvider, text: str, voice: str) -> list[bytes]:
    return [c async for c in provider.synthesize_stream(text, voice)]


async def test_hit_replays_without_calling_inner() -> None:
    inner = _SpyTTS()
    cache = CachingTTSProvider(inner, TtsCacheStore())

    first = await _drain(cache, "您好，我是助手。", "v1")
    assert len(inner.calls) == 1  # miss → inner called
    second = await _drain(cache, "您好，我是助手。", "v1")
    assert len(inner.calls) == 1  # hit → inner NOT called again
    assert second == first  # identical replay


async def test_voice_id_is_part_of_key() -> None:
    inner = _SpyTTS()
    cache = CachingTTSProvider(inner, TtsCacheStore())
    await _drain(cache, "转人工", "v1")
    await _drain(cache, "转人工", "v2")  # different voice → miss
    assert len(inner.calls) == 2


async def test_long_text_not_cached() -> None:
    inner = _SpyTTS()
    cache = CachingTTSProvider(inner, TtsCacheStore(), max_cacheable_chars=10)
    long_text = "这是一段很长的动态主回复" * 3  # > 10 chars
    await _drain(cache, long_text, "v1")
    await _drain(cache, long_text, "v1")
    assert len(inner.calls) == 2  # never cached → inner called each time


async def test_error_midstream_does_not_cache() -> None:
    inner = _SpyTTS()
    inner.fail_on = {"坏话术"}
    store = TtsCacheStore()
    cache = CachingTTSProvider(inner, store)
    with contextlib.suppress(RuntimeError):
        await _drain(cache, "坏话术", "v1")
    assert len(store) == 0  # truncated synth not cached
    # Recovered text caches normally.
    inner.fail_on = set()
    await _drain(cache, "坏话术", "v1")
    assert len(store) == 1


async def test_lru_eviction_by_entries() -> None:
    inner = _SpyTTS(chunks=1)
    store = TtsCacheStore(max_entries=2)
    cache = CachingTTSProvider(inner, store)
    await _drain(cache, "a", "v")
    await _drain(cache, "b", "v")
    await _drain(cache, "c", "v")  # evicts LRU "a"
    assert len(store) == 2
    assert store.get(("a", "v")) is None  # evicted
    assert store.get(("c", "v")) is not None


async def test_aclose_passes_through() -> None:
    inner = _SpyTTS()
    cache = CachingTTSProvider(inner, TtsCacheStore())
    await cache.aclose()
    assert inner.closed is True
