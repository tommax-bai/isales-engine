"""Keyword-driven mock LLM.

Spec: ai-pipeline § Requirement: 三层并行管线编排 (drives role / judge / polish
behaviours); role-prompt § Requirement: JSON Mode 强制策略 (we always emit
parsable JSON when ``json_mode=True``).

The orchestrator (PR #6) feeds a single ``user`` message containing the
3-section text from ``prompt_builder``; the ``system`` message identifies
whether this is a role / judge / polish call. We pattern-match on simple
markers in the system or user content to drive deterministic test fixtures
without running real LLMs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from isales_common.providers._models import LLMResponse, Message
from isales_common.providers.llm import LLMProvider


@dataclass
class _Decision:
    content: str
    tokens_in: int = 16
    tokens_out: int = 16


class KeywordDrivenMockLLM(LLMProvider):
    """Returns deterministic responses based on prompt content.

    Layer markers expected in the ``system`` message:

    * ``[role]`` — three-layer role candidate. Output JSON
      ``{"reply", "goal_achieved", "goal_type", "extracted"}``.
    * ``[judge]`` — judge call. Output JSON ``{"passed", "reason"}``.
    * ``[polish]`` — polish call. Output JSON
      ``{"reply", "selected_candidate_index"}``.
    * ``[transfer_intent]`` — intent classifier. Output JSON
      ``{"intent", "probability"}``.
    * ``[transfer_llm]`` — independent transfer LLM. Output JSON
      ``{"transfer"}``.

    Within the user-message content, additional markers override defaults:

    * ``"成功"`` / ``"预约"`` (in role calls) → ``goal_achieved=true``
      ``goal_type="appointment"``.
    * ``"请用一句话回应"`` (system message, short_reply strategy) →
      single-sentence reply.
    * ``"目标已达成"`` (system message, WRAPPING_UP) → polite goodbye reply
      with ``goal_achieved=false`` (avoids re-triggering wrap-up).
    * ``"**reject**"`` (judge user content) → ``passed=false``.
    * ``"do_not_call"`` (any user content) → role marks
      ``goal_type="do_not_call"``, ``goal_achieved=true``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], bool]] = []

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
        decision = self._decide(messages, json_mode=json_mode)
        return LLMResponse(
            content=decision.content,
            tokens_in=decision.tokens_in,
            tokens_out=decision.tokens_out,
            finish_reason="stop",
            latency_ms=0,
        )

    # ------------------------------------------------------------------
    def _decide(self, messages: list[Message], *, json_mode: bool) -> _Decision:
        system = self._first_role(messages, "system")
        user = self._first_role(messages, "user")

        # Layer dispatch based on system tags. The orchestrator stamps these
        # in PR #6 via prompt_builder.
        if "[judge]" in system:
            return self._judge(user)
        if "[polish]" in system:
            return self._polish(user)
        if "[transfer_intent]" in system:
            return self._transfer_intent(user)
        if "[transfer_llm]" in system:
            return self._transfer_llm(user)
        # Default = role call.
        return self._role(system, user, json_mode=json_mode)

    @staticmethod
    def _first_role(messages: list[Message], role: str) -> str:
        for msg in messages:
            if msg.role == role:
                return msg.content
        return ""

    # ---- Role candidates ---------------------------------------------------
    def _role(self, system: str, user: str, *, json_mode: bool) -> _Decision:
        if "do_not_call" in user:
            return _Decision(
                content=json.dumps(
                    {
                        "reply": "好的，已为您登记勿打扰，再见。",
                        "goal_achieved": True,
                        "goal_type": "do_not_call",
                        "extracted": {},
                    },
                    ensure_ascii=False,
                ),
            )

        goal_hit = bool(re.search(r"成功|预约|appointment", user))
        if goal_hit:
            content = json.dumps(
                {
                    "reply": "好的，已为您预约成功。",
                    "goal_achieved": True,
                    "goal_type": "appointment",
                    "extracted": {"appointment_time": "2026-05-07T10:00:00"},
                },
                ensure_ascii=False,
            )
            return _Decision(content=content)

        if "请用一句话回应" in system:
            content = json.dumps(
                {
                    "reply": "明白了。",
                    "goal_achieved": False,
                    "goal_type": "",
                    "extracted": {},
                },
                ensure_ascii=False,
            )
            return _Decision(content=content)

        if "目标已达成" in system:
            content = json.dumps(
                {
                    "reply": "好的，期待和您再次联系，再见。",
                    "goal_achieved": False,
                    "goal_type": "",
                    "extracted": {},
                },
                ensure_ascii=False,
            )
            return _Decision(content=content)

        # Default role reply.
        content = json.dumps(
            {
                "reply": "好的，请稍等。",
                "goal_achieved": False,
                "goal_type": "",
                "extracted": {},
            },
            ensure_ascii=False,
        )
        if not json_mode:
            # Surround with explanatory chatter so the parser must use the
            # regex fallback (role-prompt spec § Scenario "文本约束兜底").
            content = "解释：" + content + "\n（以上为输出）"
        return _Decision(content=content)

    # ---- Judge -------------------------------------------------------------
    def _judge(self, user: str) -> _Decision:
        passed = "**reject**" not in user
        reason = "ok" if passed else "rejected by mock judge"
        return _Decision(
            content=json.dumps(
                {"passed": passed, "reason": reason}, ensure_ascii=False
            ),
        )

    # ---- Polish ------------------------------------------------------------
    def _polish(self, user: str) -> _Decision:
        # User content is expected to carry candidates as
        # "candidate[0]: <reply>\ncandidate[1]: <reply>\n...".
        m = re.search(r"candidate\[(\d+)\]: (.+?)(?=\ncandidate\[|\Z)", user, re.DOTALL)
        idx = int(m.group(1)) if m else 0
        reply = (m.group(2).strip() if m else "好的。")
        return _Decision(
            content=json.dumps(
                {"reply": "好的，" + reply, "selected_candidate_index": idx},
                ensure_ascii=False,
            ),
        )

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
