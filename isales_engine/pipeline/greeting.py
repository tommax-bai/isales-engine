"""Greeting generator (固定模板 / LLM 单角色生成).

Spec: ai-pipeline § Requirement: 开场白不走管线 (no judges, no polish);
      transcript § dialog_history 与 full_transcript 双集合 (greeting goes
      into both).
"""

from __future__ import annotations

import logging

from isales_common.providers.llm import LLMProvider

from isales_engine.call_session import CallSession
from isales_engine.pipeline.json_parser import parse_role_output
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
    if not config.roles:
        return "您好。"

    role = config.roles[0]
    messages = build_greeting_messages(role, config)
    import asyncio

    try:
        async with asyncio.timeout(timeout_ms / 1000.0):
            resp = await llm.chat(
                messages,
                json_mode=True,
                temperature=role.temperature,
                top_p=role.top_p,
            )
    except Exception:  # noqa: BLE001 — greeting failure should never block dial
        logger.exception("greeting_llm_failed role_id=%s", role.role_config_id)
        return "您好。"

    parsed = parse_role_output(resp.content)
    if parsed.parse_failed or not parsed.reply:
        return "您好。"
    return parsed.reply
