"""Greeting generator (固定模板 / LLM 单角色生成).

Spec: ai-pipeline § Requirement: 开场白不走管线 (no referees);
      transcript § dialog_history 与 full_transcript 双集合 (greeting goes
      into both).

pipeline-stream-and-referee: the LLM greeting now uses the main slot's plain
text output (``chat()``, no JSON Mode / no parsing). The fixed-template path
(campaign.greeting set) still skips the LLM entirely.
"""

from __future__ import annotations

import asyncio
import logging

from isales_common.providers.llm import LLMProvider

from isales_engine.call_session import CallSession
from isales_engine.pipeline.prompt_builder import (
    PipelineConfig,
    build_greeting_messages,
)

logger = logging.getLogger(__name__)


async def generate_greeting(
    session: CallSession,
    config: PipelineConfig,
    llm: LLMProvider,
    *,
    fixed_template: str | None = None,
    timeout_ms: int = 8000,
) -> str:
    """Return the greeting text. Fixed-template path skips the LLM entirely."""

    if fixed_template:
        return fixed_template

    messages = build_greeting_messages(config)
    try:
        async with asyncio.timeout(timeout_ms / 1000.0):
            resp = await llm.chat(
                messages,
                temperature=config.main.temperature,
                top_p=config.main.top_p,
            )
    except Exception:  # noqa: BLE001 — greeting failure should never block dial
        logger.exception(
            "greeting_llm_failed role_id=%s", config.main.role_config_id
        )
        return "您好。"

    text = (resp.content or "").strip()
    return text or "您好。"
