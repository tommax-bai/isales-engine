"""Double-condition interruption detection.

Spec: interruption-detection § all Requirements.

The detector is fed ASR partial results during ``SPEAKING`` / ``FILLER`` and
emits an ``InterruptionVerdict`` per partial:

* ``ignored`` — at least one bypass condition matched (whitelist OR duration
  below threshold).
* ``triggered`` — both conditions failed; the caller MUST stop TTS / filler
  and treat the eventual ASR final as a new PROCESSING input.

Once ``triggered`` fires, subsequent partials remain ``triggered`` (verdict
is unconditional after first hit) per spec § "打断判定不可撤销".

Connected counters (consecutive interruption + reset on full SPEAKING) are
maintained on :class:`CallSession`; the detector is a stateless function plus
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["ignored", "triggered"]


@dataclass
class InterruptionConfig:
    whitelist: tuple[str, ...]
    min_duration_ms: int


@dataclass
class InterruptionVerdict:
    verdict: Verdict
    reason: str  # "whitelist" / "below_threshold" / "exceeds"


def evaluate_partial(
    *, text: str, speech_started_ts_ms: int, now_ts_ms: int, config: InterruptionConfig
) -> InterruptionVerdict:
    """Apply double-condition rule. Inputs are pre-normalised by the caller."""

    _stripped = text.strip()
    if _stripped in config.whitelist:
        return InterruptionVerdict(verdict="ignored", reason="whitelist")

    # PROTOTYPE 2026-06-03: ASR-text-length gate. ASR vendor model 不会把
    # ambient noise (椅子/风扇/呼吸/键盘) 识别成中文文字, 用 text length
    # 当 barge-in 信号 比 VAD energy 更抗噪. ≥ 2 字才算真打断, 1 字
    # (含 "对/好") 太短不区分真打断 vs 误识别. 真生产场景 user 主动
    # 打断通常是 "等等"/"不要"/"我现在" 2 字以上。
    # Hardcoded prototype value; if validated, promote to InterruptionConfig.min_text_length.
    _MIN_TEXT_LENGTH_PROTOTYPE = 2
    if len(_stripped) < _MIN_TEXT_LENGTH_PROTOTYPE:
        return InterruptionVerdict(verdict="ignored", reason="text_too_short")

    elapsed = now_ts_ms - speech_started_ts_ms
    if elapsed < config.min_duration_ms:
        return InterruptionVerdict(verdict="ignored", reason="below_threshold")

    return InterruptionVerdict(verdict="triggered", reason="exceeds")
