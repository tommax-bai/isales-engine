"""Split a main-LLM token stream into TTS-ready sentences.

Spec: ai-pipeline § "sentence boundary detector".

Rule (mirrors voxen textproc/processor.go):
  - accumulate tokens into a buffer
  - emit a sentence when the buffer ends on a sentence terminator
    (``。？！.?!`` or ``\n\n``)
  - OR, once the buffer reaches a *soft* length, emit at the next *soft*
    boundary (pause punctuation ``，、；：,;:``) — a natural clause break
  - OR, as a last-resort fallback, emit at a *hard* ceiling so a punctuation-
    less run-on doesn't hold first audio back indefinitely
  - flush whatever remains when the stream ends

Length-based splitting was originally a hard cut at exactly 50 chars regardless
of position (``len(buffer) >= 50``), which sliced mid-clause — the single
ugliest source of cross-sentence prosody jumps (each fragment is its own cold
TTS request, so cutting mid-word both makes the splice audible and multiplies
the per-request prosody resets). engine-sentence-splitter-soft-boundary
replaces that with: don't let length cut at all below ``SOFT_SENTENCE_CHARS``;
above it, defer the cut to the next pause-punctuation boundary so we break where
prosody already dips; only fall back to a hard cut at ``MAX_SENTENCE_CHARS``
(120) for the rare punctuation-less run-on. Splitting on already-decoded ``str``
chunks means we never cut a multi-byte char.
"""

from __future__ import annotations

import unicodedata
from collections.abc import AsyncIterator

# Sentence-ending punctuation (Chinese + ASCII). A trailing closing quote /
# bracket after a terminator is kept with the sentence it closes.
_TERMINATORS = "。？！?!."
_TRAILING = "”’\"')）」』】"

# Pause / clause punctuation (Chinese + ASCII). Once the buffer is long enough
# (>= SOFT_SENTENCE_CHARS) a cut is allowed here — a natural prosodic dip — so a
# long sentence is broken at a clause boundary instead of mid-word.
_SOFT_BOUNDARIES = "，、；：,;:"

# Soft length: below this, length never forces a cut (short sentences are
# untouched — they only split on a real terminator). At/above it, the next soft
# boundary cuts.
SOFT_SENTENCE_CHARS = 60

# Hard ceiling (last-resort fallback for a punctuation-less run-on). Removal
# trigger: only here to bound first-audio latency for malformed/no-punctuation
# LLM output; if the soft-boundary path proves sufficient in production this can
# be raised further or dropped. Was 50 (the old hard mid-clause cut).
MAX_SENTENCE_CHARS = 120


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
    soft_chars: int = SOFT_SENTENCE_CHARS,
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
            # Emit on: a sentence terminator; a blank-line break; once long
            # enough (>= soft_chars) the next pause-punctuation boundary so we
            # cut at a natural clause break rather than mid-word; or the hard
            # ceiling as a last-resort fallback for a punctuation-less run-on.
            should_emit = (
                ch in _TERMINATORS
                or buffer.endswith("\n\n")
                or (len(buffer) >= soft_chars and ch in _SOFT_BOUNDARIES)
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
__all__ = [
    "split_sentences",
    "SOFT_SENTENCE_CHARS",
    "MAX_SENTENCE_CHARS",
    "_SOFT_BOUNDARIES",
    "_TRAILING",
]
