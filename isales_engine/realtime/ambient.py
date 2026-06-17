"""Continuous outbound background-ambient mixing (engine-ambient-background-mix).

Loads a preprocessed background-noise asset to the engine's outbound format
(16 kHz mono int16) and serves it as an endless, click-free loop for the
outbound mixing pump in :mod:`rtc_telephony`. This module owns only asset
decode + seamless looping + sample-wise mix; the 10 ms frame grid and the push
to DingRTC stay in the telephony layer.

ISOLATION INVARIANT (ambient-background-mix spec): background audio is mixed
ONLY on the outbound (engine → customer) path. Nothing here touches the inbound
ASR / 断句 stream — background never enters the audio fed to ASR.

Assets live under ``$ISALES_AMBIENT_DIR`` (default ``/opt/isales/ambient``); a
campaign's ``ambient_audio`` column is the asset basename. Decode is best-effort:
a missing / unreadable asset logs and yields no source, so the call proceeds on
the normal (background-off) outbound path rather than failing — this is graceful
degradation of an optional feature, not a fallback layer.
"""

from __future__ import annotations

import logging
import os
import wave
from array import array
from pathlib import Path

try:  # Py 3.13+ removes audioop; engine still targets 3.12 (see _inbound_loop).
    import audioop
except ImportError:  # pragma: no cover
    audioop = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Engine outbound format — must match DingRtcSession send rate/channels.
SEND_RATE = 16000
SEND_CHANNELS = 1
SAMPLE_BYTES = 2
FRAME_SAMPLES = SEND_RATE // 100  # 160 samples = 10 ms
FRAME_BYTES = FRAME_SAMPLES * SEND_CHANNELS * SAMPLE_BYTES  # 320

SILENCE_FRAME = b"\x00" * FRAME_BYTES

_DEFAULT_DIR = "/opt/isales/ambient"
_CROSSFADE_SAMPLES = SEND_RATE // 50  # 20 ms seam crossfade


def _asset_dir() -> Path:
    return Path(os.environ.get("ISALES_AMBIENT_DIR", _DEFAULT_DIR))


def _clip16(v: float) -> int:
    return int(-32768 if v < -32768 else 32767 if v > 32767 else v)


def mix_frame(tts: bytes, bg: bytes, gain: float) -> bytes:
    """Mix one TTS frame with one background frame (pure int16, no audioop).

    ``bg`` is scaled by ``gain`` (linear, 0.1 ≈ -20dB) then added to ``tts`` with
    int16 saturation. Pure ``array`` arithmetic so mixing works on every Python
    (audioop was removed in 3.13) and is unit-testable without a vendored
    backport. With gain ≤ 0 or empty ``bg``, returns ``tts`` unchanged.

    Little-endian int16 (all engine deploy targets — Linux x86_64 / mac arm —
    are LE, matching the ``array('h')`` native layout used elsewhere).
    """
    if gain <= 0 or not bg:
        return tts
    st = array("h")
    st.frombytes(tts)
    sb = array("h")
    sb.frombytes(bg)
    n = min(len(st), len(sb))
    for i in range(n):
        s = st[i] + sb[i] * gain
        if s > 32767 or s < -32768:
            st[i] = _clip16(s)
        else:
            st[i] = int(s)
    return st.tobytes()


def apply_fadeout(frames: list[bytes]) -> list[bytes]:
    """Ramp ``frames`` down to silence with a per-sample linear gain fade.

    engine-barge-in-fade-out: on barge-in the playout queue is no longer hard-
    cut to zero (an amplitude step at a frame boundary = audible click + a
    faster-than-human stop). Instead the first frames still buffered are ramped
    from gain 1.0 (first sample) to 0.0 (last sample) across the whole supplied
    span, then re-queued, so the AI voice trails off smoothly. The fade is
    per-sample (not per-frame) so there is no inter-frame zipper, and it lands
    on exact silence so nothing clicks after it. Pure ``array('h')`` like
    :func:`mix_frame` (no audioop). Input is not mutated; new buffers returned.
    """
    total = sum(len(f) // SAMPLE_BYTES for f in frames)
    if total <= 1:
        return list(frames)
    out: list[bytes] = []
    idx = 0
    for f in frames:
        sa = array("h")
        sa.frombytes(f)
        for j in range(len(sa)):
            sa[j] = _clip16(sa[j] * (1.0 - idx / (total - 1)))
            idx += 1
        out.append(sa.tobytes())
    return out


def _linear_crossfade(a: bytes, b: bytes) -> bytes:
    """Per-sample linear crossfade a→b (a fades out, b fades in)."""
    sa = array("h")
    sa.frombytes(a)
    sb = array("h")
    sb.frombytes(b)
    n = min(len(sa), len(sb))
    out = array("h", bytes(n * SAMPLE_BYTES))
    for i in range(n):
        t = (i + 1) / (n + 1)
        out[i] = _clip16(sa[i] * (1.0 - t) + sb[i] * t)
    return out.tobytes()


def _make_seamless(pcm: bytes) -> bytes:
    """Pre-render a click-free loop buffer.

    Crossfade the tail into the head once at load time so a reader can read
    linearly and wrap at the buffer boundary with no audible pop: the seam's end
    matches the body's start and the body's end matches the seam's start.
    Returns whole-frame-aligned bytes.
    """
    usable = (len(pcm) // FRAME_BYTES) * FRAME_BYTES
    pcm = pcm[:usable]
    cf_bytes = _CROSSFADE_SAMPLES * SAMPLE_BYTES
    if len(pcm) <= 2 * cf_bytes:
        return pcm  # too short to crossfade — read as-is
    head = pcm[:cf_bytes]
    tail = pcm[-cf_bytes:]
    body = pcm[cf_bytes:-cf_bytes]
    seam = _linear_crossfade(tail, head)
    loop = body + seam
    usable = (len(loop) // FRAME_BYTES) * FRAME_BYTES
    return loop[:usable]


class AmbientLoop:
    """Immutable, process-cached seamless loop buffer (shared across calls)."""

    def __init__(self, pcm: bytes, *, name: str) -> None:
        self._pcm = pcm
        self.name = name

    def __bool__(self) -> bool:
        return len(self._pcm) >= FRAME_BYTES

    def reader(self) -> AmbientReader:
        return AmbientReader(self._pcm)


class AmbientReader:
    """Per-call cursor over a shared :class:`AmbientLoop` buffer."""

    def __init__(self, pcm: bytes) -> None:
        self._pcm = pcm
        self._pos = 0  # frame-aligned byte offset

    def next_frame(self) -> bytes:
        n = len(self._pcm)
        if n < FRAME_BYTES:
            return SILENCE_FRAME
        start = self._pos
        end = start + FRAME_BYTES
        if end > n:  # buffer is frame-aligned, but guard anyway
            start, end = 0, FRAME_BYTES
        frame = self._pcm[start:end]
        self._pos = end if end < n else 0
        return frame


_loop_cache: dict[str, AmbientLoop | None] = {}


def get_ambient_loop(name: str | None) -> AmbientLoop | None:
    """Resolve + decode a background asset to a cached seamless loop.

    Returns ``None`` (and logs once) for empty name, missing/unreadable file,
    decode error, or absent audioop — the caller then keeps the background-off
    outbound path.
    """
    if not name:
        return None
    if name in _loop_cache:
        return _loop_cache[name]
    loop = _load(name)
    _loop_cache[name] = loop
    return loop


def _load(name: str) -> AmbientLoop | None:
    # Reject path traversal — asset id is a basename under the asset dir.
    if "/" in name or "\\" in name or name in (".", ".."):
        logger.warning("ambient_load_rejected name=%s reason=not_a_basename", name)
        return None
    path = _asset_dir() / name
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
    except (OSError, wave.Error) as exc:
        logger.warning("ambient_load_failed name=%s path=%s err=%s", name, path, exc)
        return None
    if not pcm:
        logger.warning("ambient_load_empty name=%s path=%s", name, path)
        return None
    # Assets SHOULD be preprocessed to 16 kHz mono int16 (design D3) so no
    # conversion is needed and loading works without audioop. If a non-target
    # asset is supplied, convert via audioop when present; without it (Python
    # 3.13+ removed audioop), skip the asset rather than ship a wrong-rate loop.
    if width != SAMPLE_BYTES or channels > 1 or rate != SEND_RATE:
        if audioop is None:
            logger.warning(
                "ambient_load_needs_conversion name=%s rate=%s ch=%s width=%s "
                "but audioop unavailable — preprocess to 16kHz mono int16",
                name, rate, channels, width,
            )
            return None
        if width != SAMPLE_BYTES:
            pcm = audioop.lin2lin(pcm, width, SAMPLE_BYTES)
        if channels > 1:
            pcm = audioop.tomono(pcm, SAMPLE_BYTES, 0.5, 0.5)
        if rate != SEND_RATE:
            pcm, _ = audioop.ratecv(pcm, SAMPLE_BYTES, 1, rate, SEND_RATE, None)
    loop = AmbientLoop(_make_seamless(pcm), name=name)
    if not loop:
        logger.warning("ambient_load_too_short name=%s path=%s", name, path)
        return None
    logger.info(
        "ambient_loaded name=%s path=%s src_rate=%s src_ch=%s loop_ms=%.0f",
        name, path, rate, channels,
        len(loop._pcm) / FRAME_BYTES * 10.0,
    )
    return loop
