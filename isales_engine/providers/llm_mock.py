"""Keyword-driven mock LLM (dual-LLM architecture).

pipeline-stream-and-referee: the mock now models the two-LLM pipeline:

* ``chat_stream`` is the **main** path → yields **plain text** (no JSON), token
  by token, so the sentence splitter + TTS path can be exercised.
* ``chat(json_mode=True)`` is used by the **referee** + transfer classifiers →
  emits JSON. Referee dispatch keys off the referee user-primer text; transfer
  classifiers off ``[transfer_intent]`` / ``[transfer_llm]`` system markers.
* ``chat`` without those markers is the greeting path → plain text.

Keyword triggers (referee, matched on the substituted system prompt which
carries ``user_last_utterance``):
  * ``预约`` / ``成功`` / ``约见`` → ``goal_achieved`` (``appointment``)
  * ``转人工`` / ``人工`` → ``transfer``
  * ``拒绝`` / ``不需要`` / ``没兴趣`` / ``do_not_call`` → ``customer_decline``
  * otherwise → ``continue``
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider

# Must match isales_engine.referee._USER_PRIMER (referee user message).
_REFEREE_PRIMER_MARK = "JSON schema 输出"


@dataclass
class _Decision:
    content: str
    tokens_in: int = 16
    tokens_out: int = 16


class KeywordDrivenMockLLM(LLMProvider):
    """Deterministic dual-LLM mock for engine tests."""

    def __init__(
        self,
        *,
        first_token_ms: float = 0.0,
        per_token_ms: float = 0.0,
    ) -> None:
        self.calls: list[tuple[list[Message], bool]] = []
        self._first_token_s = first_token_ms / 1000.0
        self._per_token_s = per_token_ms / 1000.0

    # ---- chat (referee / transfer classifiers / greeting) -----------------
    async def chat(
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), json_mode))
        decision = self._decide_chat(messages)
        return LLMResponse(
            content=decision.content,
            tokens_in=decision.tokens_in,
            tokens_out=decision.tokens_out,
            finish_reason="stop",
            latency_ms=0,
        )

    # ---- chat_stream (main, plain text) -----------------------------------
    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.calls.append((list(messages), False))
        text = self._decide_main(messages)
        first = True
        for ch in text:
            if first:
                if self._first_token_s:
                    await asyncio.sleep(self._first_token_s)
                first = False
            elif self._per_token_s:
                await asyncio.sleep(self._per_token_s)
            yield ch
        self.last_call_tokens_in = 16
        self.last_call_tokens_out = max(1, len(text) // 2)
        self.last_call_finish_reason = "stop"

    # ------------------------------------------------------------------
    @staticmethod
    def _first_role(messages: list[Message], role: str) -> str:
        for msg in messages:
            if msg.role == role:
                return msg.content
        return ""

    def _decide_chat(self, messages: list[Message]) -> _Decision:
        system = self._first_role(messages, "system")
        user = self._first_role(messages, "user")
        if "[transfer_intent]" in system:
            return self._transfer_intent(user)
        if "[transfer_llm]" in system:
            return self._transfer_llm(user)
        if _REFEREE_PRIMER_MARK in user:
            return self._referee(system)
        # Greeting / other plain-text chat.
        return _Decision(content="您好，我是 AI 助手，请问现在方便吗？")

    # ---- Main (plain text reply) ------------------------------------------
    def _decide_main(self, messages: list[Message]) -> str:
        system = self._first_role(messages, "system")
        if "目标已达成" in system or "收尾对话" in system:
            return "好的，期待和您再次联系，再见。"
        if "请用一句话回应" in system:
            return "明白了。"
        return "好的，我明白了。请问您还有什么需要？"

    # ---- Referee (JSON {category, confidence}) ----------------------------
    # engine-multi-referee-and-restructure: a referee emits a free category
    # string (not a strong decision enum). The categories below match the
    # backward-compatible default routing rules seeded by the migration
    # (goal_achieved / transfer / customer_decline; "continue" matches no rule).
    def _referee(self, system: str) -> _Decision:
        if re.search(r"预约|成功|约见|appointment", system):
            payload = {"category": "goal_achieved", "confidence": 0.95}
        elif re.search(r"转人工|人工", system):
            payload = {"category": "transfer", "confidence": 0.9}
        elif re.search(r"拒绝|不需要|没兴趣|do_not_call", system):
            payload = {"category": "customer_decline", "confidence": 0.9}
        else:
            payload = {"category": "continue", "confidence": 0.9}
        return _Decision(content=json.dumps(payload, ensure_ascii=False))

    # ---- Transfer intent ---------------------------------------------------
    def _transfer_intent(self, user: str) -> _Decision:
        prob = 0.95 if "转人工" in user or "人工" in user else 0.1
        return _Decision(
            content=json.dumps(
                {"intent": "transfer", "probability": prob}, ensure_ascii=False
            ),
        )

    # ---- Transfer LLM (independent) ---------------------------------------
    def _transfer_llm(self, user: str) -> _Decision:
        transfer = "投诉" in user or "退款" in user
        return _Decision(
            content=json.dumps({"transfer": transfer}, ensure_ascii=False),
        )
