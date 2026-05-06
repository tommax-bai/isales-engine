"""Human-handoff trigger evaluation + TRANSFERRING flow.

Spec: human-handoff § all Requirements.

Four independent OR-relation triggers:

* ``keyword`` — ASR final text contains any of ``transfer_keywords``.
* ``intent`` — intent classifier (LLM) output ``probability`` exceeds
  ``transfer_intent_threshold``.
* ``round`` — dialog turn count exceeds ``transfer_round_threshold`` AND
  ``goal_achieved`` is False.
* ``llm`` — independent LLM returns ``{"transfer": true}``.

The TRANSFERRING flow itself (random transfer phrase → TTS → hangup → write
``call_record.transfer_status``) is driven by ``CallSession.run()`` in
PR #11; this module supplies the trigger detector + a config dataclass.

handoff_task creation is the worker's job (impl-worker stage 3B already
ships that on receiving CallEnded with ``transfer_status='marked_for_handoff'``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass

from isales_common.providers._models import Message
from isales_common.providers.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    keyword_enabled: bool
    keywords: tuple[str, ...]
    intent_enabled: bool
    intent_threshold: float | None
    intent_system_prompt: str
    round_enabled: bool
    round_threshold: int | None
    llm_enabled: bool
    llm_system_prompt: str
    phrases: tuple[str, ...]


@dataclass
class TransferDecision:
    triggered: bool
    trigger_type: str | None  # "keyword" / "intent" / "round" / "llm" / None
    trigger_detail: str
    phrase: str  # selected hand-off phrase (random pick, empty if not triggered)


async def evaluate_transfer(
    *,
    user_text: str,
    turn_count: int,
    goal_achieved: bool,
    config: TransferConfig,
    llm: LLMProvider,
    intent_timeout_s: float = 5.0,
    llm_timeout_s: float = 5.0,
) -> TransferDecision:
    """OR-relation evaluation of all four triggers; first hit wins."""

    # 1) keyword (cheap, sync)
    if config.keyword_enabled and config.keywords:
        for kw in config.keywords:
            if kw and kw in user_text:
                return _hit("keyword", kw, config)

    # 2) round threshold (cheap, sync)
    if (
        config.round_enabled
        and config.round_threshold is not None
        and turn_count >= config.round_threshold
        and not goal_achieved
    ):
        return _hit("round", f"turn_count={turn_count}", config)

    # 3) intent (LLM call)
    if config.intent_enabled and config.intent_threshold is not None:
        prob = await _call_intent(
            user_text, config.intent_system_prompt, llm, timeout_s=intent_timeout_s
        )
        if prob is not None and prob >= config.intent_threshold:
            return _hit("intent", f"probability={prob:.2f}", config)

    # 4) independent LLM
    if config.llm_enabled:
        flag = await _call_independent_llm(
            user_text, config.llm_system_prompt, llm, timeout_s=llm_timeout_s
        )
        if flag:
            return _hit("llm", "transfer=true", config)

    return TransferDecision(
        triggered=False, trigger_type=None, trigger_detail="", phrase=""
    )


def _hit(trigger_type: str, detail: str, config: TransferConfig) -> TransferDecision:
    phrase = random.choice(config.phrases) if config.phrases else ""
    return TransferDecision(
        triggered=True, trigger_type=trigger_type, trigger_detail=detail, phrase=phrase
    )


async def _call_intent(
    user_text: str, system_prompt: str, llm: LLMProvider, *, timeout_s: float
) -> float | None:
    messages = [
        Message(role="system", content="[transfer_intent] " + system_prompt),
        Message(role="user", content=user_text),
    ]
    try:
        async with asyncio.timeout(timeout_s):
            resp = await llm.chat(messages, json_mode=True)
    except (TimeoutError, Exception):  # noqa: BLE001
        logger.exception("transfer_intent_failed")
        return None

    try:
        parsed = json.loads(resp.content)
    except (ValueError, TypeError):
        return None
    prob = parsed.get("probability")
    return float(prob) if isinstance(prob, int | float) else None


async def _call_independent_llm(
    user_text: str, system_prompt: str, llm: LLMProvider, *, timeout_s: float
) -> bool:
    messages = [
        Message(role="system", content="[transfer_llm] " + system_prompt),
        Message(role="user", content=user_text),
    ]
    try:
        async with asyncio.timeout(timeout_s):
            resp = await llm.chat(messages, json_mode=True)
    except (TimeoutError, Exception):  # noqa: BLE001
        logger.exception("transfer_llm_failed")
        return False

    try:
        parsed = json.loads(resp.content)
    except (ValueError, TypeError):
        return False
    return bool(parsed.get("transfer", False))
