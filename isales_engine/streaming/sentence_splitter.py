"""Split a main-LLM token stream into TTS-ready sentences.

Spec: ai-pipeline § "单 main LLM streaming"; design.md 决策 8.

Rule (mirrors voxen textproc/processor.go):
  - accumulate tokens into a buffer
  - emit a sentence when the buffer ends on a sentence terminator
    (``。？！.?!`` or ``\n\n``) OR exceeds ``MAX_SENTENCE_CHARS``
  - flush whatever remains when the stream ends

The 50-char cap keeps a single TTS synthesize call short so first audio is not
held back by one very long sentence. Splitting at the cap avoids cutting a
multi-byte char because we accumulate already-decoded ``str`` chunks.
"""

from __future__ import annotations

import unicodedata
from collections.abc import AsyncIterator

# Sentence-ending punctuation (Chinese + ASCII). A trailing closing quote /
# bracket after a terminator is kept with the sentence it closes.
_TERMINATORS = "。？！?!."
_TRAILING = "”’\"')）」』】"

MAX_SENTENCE_CHARS = 50


def _has_synthesizable_text(s: str) -> bool:
    """True if ``s`` has ≥1 character TTS can voice (a letter or a digit).

    Filters chunks that are punctuation / whitespace / symbol only — e.g. a
    bare "." emitted when an ellipsis "..." is split on each dot, which the
    Volcengine TTS vendor rejects with ``code=45002001 "No readable text!"``
    (call 194 turn 4). Uses Unicode general categories rather than a hardcoded
    punctuation table: any L* (letter, incl. CJK ideographs like 呃) or N*
    (number) char makes the chunk synthesizable; P*/Z*/S*/C* on their own do
    not. Spec: ai-pipeline § "TTS 不合成无可读内容的句".
    """
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in s)


async def split_sentences(
    token_stream: AsyncIterator[str],
    *,
    max_chars: int = MAX_SENTENCE_CHARS,
) -> AsyncIterator[str]:
    """Yield sentences from ``token_stream``.

    Each yielded string is stripped of leading/trailing whitespace and is
    non-empty. Tokens may be arbitrarily fragmented ("您"+"好"+"。"): the
    splitter buffers across token boundaries.
    """
    buffer = ""
    async for token in token_stream:
        if not token:
            continue
        for ch in token:
            buffer += ch
            # Emit on a sentence terminator, a blank-line break, or the char
            # cap (keeps a single TTS synth short so first audio isn't held by
            # one long sentence).
            should_emit = (
                ch in _TERMINATORS
                or buffer.endswith("\n\n")
                or len(buffer) >= max_chars
            )
            if should_emit:
                out = buffer.strip()
                if out and _has_synthesizable_text(out):
                    yield out
                buffer = ""

    tail = buffer.strip()
    if tail and _has_synthesizable_text(tail):
        yield tail


async def _stream_from(parts: list[str]) -> AsyncIterator[str]:
    """Helper for tests / callers that already have the full token list."""
    for p in parts:
        yield p


# Keep the trailing-punctuation set importable for callers / tests.
__all__ = ["split_sentences", "MAX_SENTENCE_CHARS", "_TRAILING"]
