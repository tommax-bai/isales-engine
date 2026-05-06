"""Three-layer parallel AI pipeline orchestrator.

Spec: ai-pipeline § all Requirements; goal-achievement § 三层管线对结构化标记的处理;
      role-prompt § JSON Mode 强制策略; impl-engine spec delta § per-turn
      pipeline_trace.

One ``run_pipeline()`` call drives one PROCESSING turn:

1. Layer 1 — N parallel role LLM calls (gathered with return_exceptions).
2. Drop candidates that failed JSON parsing or raised provider errors.
3. Layer 2 — for each surviving candidate, call M judges in parallel.
   Any judge marking ``passed=False`` drops that candidate.
4. Layer 3 — polish LLM picks one of the passing candidates and rewrites the
   reply. Polish failure (timeout / non-JSON) falls back to the first
   passing candidate.
5. If no candidates pass (parse-fail + judge-rejection are equivalent), pick
   a random ``default_reply`` and emit a ``default_reply_used`` transcript
   event.

Goal markers (``goal_achieved`` / ``goal_type`` / ``extracted``) are inherited
unchanged from the *selected* role candidate — never voted across candidates.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from isales_common.providers.llm import LLMProvider

from isales_engine.call_session import CallSession
from isales_engine.pipeline.json_parser import (
    parse_judge_output,
    parse_polish_output,
    parse_role_output,
)
from isales_engine.pipeline.prompt_builder import (
    JudgeSpec,
    PipelineConfig,
    PolishSpec,
    RoleSpec,
    build_judge_messages,
    build_polish_messages,
    build_role_messages,
)
from isales_engine.transcript_recorder import now_utc

logger = logging.getLogger(__name__)

OutcomeSource = Literal[
    "polished", "polish_fallback", "default_reply", "wrap_up_simplified"
]


@dataclass
class PipelineOutcome:
    reply: str
    goal_achieved: bool
    goal_type: str
    extracted: dict[str, Any]
    selected_candidate_index: int
    source: OutcomeSource
    error: str | None = None


@dataclass
class _Candidate:
    role: RoleSpec
    raw_output: str
    parsed_reply: str
    parsed_goal_achieved: bool
    parsed_goal_type: str
    parsed_extracted: dict[str, Any]
    duration_ms: int
    tokens_in: int
    tokens_out: int
    error: str | None = None
    parse_failed: bool = False
    judge_results: list[dict[str, Any]] = field(default_factory=list)
    judge_passed: bool = False


async def run_pipeline(
    session: CallSession,
    user_input: str,
    config: PipelineConfig,
    llm: LLMProvider,
    *,
    is_wrap_up: bool = False,
    pipeline_timeout_ms: int = 8000,
) -> PipelineOutcome:
    """Run one full pipeline turn and append a pipeline_trace record."""

    session.current_turn_id += 1
    turn_id = session.current_turn_id
    ts_start = now_utc()
    trace_record: dict[str, Any] = {
        "turn_id": turn_id,
        "ts_start": ts_start,
        "ts_end": None,
        "user_input": user_input,
        "role_candidates": [],
        "judge_results": [],
        "polish_input": None,
        "polish_output": None,
        "polish_duration_ms": None,
        "polish_role_config_id": config.polish.role_config_id,
        "polish_prompt_version_id": config.polish.prompt_version_id,
        "final_selected_candidate_index": -1,
    }

    try:
        outcome = await _run_layers(
            session,
            user_input,
            config,
            llm,
            is_wrap_up=is_wrap_up,
            pipeline_timeout_ms=pipeline_timeout_ms,
            trace=trace_record,
        )
    except Exception as exc:  # noqa: BLE001 — trace must capture all errors
        logger.exception("pipeline_unexpected_error turn_id=%s", turn_id)
        trace_record["polish_input"] = {"pipeline_error": str(exc)}
        outcome = _default_reply_outcome(
            session, config, error=f"pipeline_unexpected: {exc}"
        )
    finally:
        trace_record["ts_end"] = now_utc()
        session.pipeline_trace_records.append(trace_record)

    return outcome


async def _run_layers(
    session: CallSession,
    user_input: str,
    config: PipelineConfig,
    llm: LLMProvider,
    *,
    is_wrap_up: bool,
    pipeline_timeout_ms: int,
    trace: dict[str, Any],
) -> PipelineOutcome:
    timeout_s = pipeline_timeout_ms / 1000.0

    # ---- Layer 1: role candidates (PK or single for wrap-up) ----
    roles = config.roles if not is_wrap_up else config.roles[:1]
    if not roles:
        return _default_reply_outcome(
            session, config, error="no_role_configured", trace=trace
        )

    candidates = await _call_roles_parallel(
        session, user_input, roles, config, llm, timeout_s=timeout_s, is_wrap_up=is_wrap_up
    )
    trace["role_candidates"] = [_serialize_candidate(c) for c in candidates]

    surviving = [c for c in candidates if not c.parse_failed and c.error is None]
    if not surviving:
        return _default_reply_outcome(
            session, config, error="all_roles_failed_or_parse", trace=trace
        )

    # ---- Layer 2: judges (skipped during wrap-up) ----
    if not is_wrap_up and config.judges:
        await _run_judges_parallel(
            surviving, config.judges, llm, timeout_s=timeout_s
        )
        trace["judge_results"] = [
            judge for c in surviving for judge in c.judge_results
        ]
        passing = [c for c in surviving if c.judge_passed]
        if not passing:
            return _default_reply_outcome(
                session, config, error="all_judges_rejected", trace=trace
            )
    else:
        for c in surviving:
            c.judge_passed = True
        passing = surviving

    # ---- Layer 3: polish ----
    polish_input = {
        "candidates": [c.parsed_reply for c in passing],
        "polish_role_config_id": config.polish.role_config_id,
    }
    trace["polish_input"] = polish_input
    polish_start = time.monotonic()
    polish_reply, polish_idx, polish_raw, polish_err = await _call_polish(
        config.polish, [c.parsed_reply for c in passing], llm, timeout_s=timeout_s
    )
    polish_duration_ms = int((time.monotonic() - polish_start) * 1000)
    trace["polish_duration_ms"] = polish_duration_ms

    if polish_reply is None or polish_idx is None or polish_idx >= len(passing):
        # Polish failure → first passing candidate.
        chosen = passing[0]
        global_index = candidates.index(chosen)
        trace["polish_output"] = polish_raw
        trace["final_selected_candidate_index"] = global_index
        return PipelineOutcome(
            reply=chosen.parsed_reply,
            goal_achieved=chosen.parsed_goal_achieved,
            goal_type=chosen.parsed_goal_type,
            extracted=chosen.parsed_extracted,
            selected_candidate_index=global_index,
            source="polish_fallback" if not is_wrap_up else "wrap_up_simplified",
            error=polish_err,
        )

    chosen = passing[polish_idx]
    global_index = candidates.index(chosen)
    trace["polish_output"] = polish_reply
    trace["final_selected_candidate_index"] = global_index

    return PipelineOutcome(
        reply=polish_reply,
        goal_achieved=chosen.parsed_goal_achieved,
        goal_type=chosen.parsed_goal_type,
        extracted=chosen.parsed_extracted,
        selected_candidate_index=global_index,
        source="polished" if not is_wrap_up else "wrap_up_simplified",
    )


async def _call_roles_parallel(
    session: CallSession,
    user_input: str,
    roles: list[RoleSpec],
    config: PipelineConfig,
    llm: LLMProvider,
    *,
    timeout_s: float,
    is_wrap_up: bool,
) -> list[_Candidate]:
    async def _call_one(role: RoleSpec) -> _Candidate:
        messages = build_role_messages(session, role, config, is_wrap_up=is_wrap_up)
        start = time.monotonic()
        try:
            async with asyncio.timeout(timeout_s):
                resp = await llm.chat(
                    messages,
                    json_mode=True,
                    temperature=role.temperature,
                    top_p=role.top_p,
                )
        except TimeoutError:
            return _Candidate(
                role=role,
                raw_output="",
                parsed_reply="",
                parsed_goal_achieved=False,
                parsed_goal_type="",
                parsed_extracted={},
                duration_ms=int((time.monotonic() - start) * 1000),
                tokens_in=0,
                tokens_out=0,
                error="timeout",
                parse_failed=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _Candidate(
                role=role,
                raw_output="",
                parsed_reply="",
                parsed_goal_achieved=False,
                parsed_goal_type="",
                parsed_extracted={},
                duration_ms=int((time.monotonic() - start) * 1000),
                tokens_in=0,
                tokens_out=0,
                error=f"provider_error: {exc}",
                parse_failed=True,
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        parsed = parse_role_output(resp.content)
        if parsed.parse_failed:
            return _Candidate(
                role=role,
                raw_output=resp.content,
                parsed_reply="",
                parsed_goal_achieved=False,
                parsed_goal_type="",
                parsed_extracted={},
                duration_ms=duration_ms,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                parse_failed=True,
            )
        return _Candidate(
            role=role,
            raw_output=resp.content,
            parsed_reply=parsed.reply,
            parsed_goal_achieved=parsed.goal_achieved,
            parsed_goal_type=parsed.goal_type,
            parsed_extracted=parsed.extracted,
            duration_ms=duration_ms,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
        )

    return list(await asyncio.gather(*(_call_one(r) for r in roles)))


async def _run_judges_parallel(
    candidates: list[_Candidate],
    judges: list[JudgeSpec],
    llm: LLMProvider,
    *,
    timeout_s: float,
) -> None:
    """Mutates each candidate in-place: fills ``judge_results`` + ``judge_passed``."""

    async def _call_judge(
        candidate_index: int, judge: JudgeSpec, reply: str
    ) -> dict[str, Any]:
        messages = build_judge_messages(judge, reply)
        start = time.monotonic()
        try:
            async with asyncio.timeout(timeout_s):
                resp = await llm.chat(messages, json_mode=True)
        except TimeoutError:
            return {
                "candidate_index": candidate_index,
                "role_config_id": judge.role_config_id,
                "prompt_version_id": judge.prompt_version_id,
                "passed": False,
                "reason": "timeout",
                "duration_ms": int((time.monotonic() - start) * 1000),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "candidate_index": candidate_index,
                "role_config_id": judge.role_config_id,
                "prompt_version_id": judge.prompt_version_id,
                "passed": False,
                "reason": f"provider_error: {exc}",
                "duration_ms": int((time.monotonic() - start) * 1000),
            }
        passed, reason = parse_judge_output(resp.content)
        return {
            "candidate_index": candidate_index,
            "role_config_id": judge.role_config_id,
            "prompt_version_id": judge.prompt_version_id,
            "passed": passed,
            "reason": reason,
            "duration_ms": int((time.monotonic() - start) * 1000),
        }

    tasks: list[asyncio.Task[dict[str, Any]]] = []
    indexed: list[tuple[int, JudgeSpec]] = []
    for i, candidate in enumerate(candidates):
        for judge in judges:
            tasks.append(
                asyncio.create_task(_call_judge(i, judge, candidate.parsed_reply))
            )
            indexed.append((i, judge))

    results = await asyncio.gather(*tasks, return_exceptions=False)
    by_candidate: dict[int, list[dict[str, Any]]] = {
        i: [] for i in range(len(candidates))
    }
    for r in results:
        by_candidate[r["candidate_index"]].append(r)

    for i, candidate in enumerate(candidates):
        candidate.judge_results = by_candidate[i]
        candidate.judge_passed = all(j["passed"] for j in candidate.judge_results)


async def _call_polish(
    polish: PolishSpec,
    candidates: list[str],
    llm: LLMProvider,
    *,
    timeout_s: float,
) -> tuple[str | None, int | None, str | None, str | None]:
    messages = build_polish_messages(polish, candidates)
    try:
        async with asyncio.timeout(timeout_s):
            resp = await llm.chat(messages, json_mode=True)
    except TimeoutError:
        return None, None, None, "polish_timeout"
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"polish_provider_error: {exc}"

    reply, idx = parse_polish_output(resp.content)
    if reply is None or idx is None:
        return None, None, resp.content, "polish_parse_failed"
    return reply, idx, resp.content, None


def _default_reply_outcome(
    session: CallSession,
    config: PipelineConfig,
    *,
    error: str,
    trace: dict[str, Any] | None = None,
) -> PipelineOutcome:
    reply = (
        random.choice(config.default_replies)
        if config.default_replies
        else "好的，请稍等。"
    )
    session.append_event("default_reply_used", text=reply, reason=error)
    if trace is not None:
        trace["polish_input"] = {"pipeline_error": error}
    return PipelineOutcome(
        reply=reply,
        goal_achieved=False,
        goal_type="",
        extracted={},
        selected_candidate_index=-1,
        source="default_reply",
        error=error,
    )


def _serialize_candidate(c: _Candidate) -> dict[str, Any]:
    return {
        "role_config_id": c.role.role_config_id,
        "prompt_version_id": c.role.prompt_version_id,
        "raw_output": c.raw_output,
        "parsed_json": {
            "reply": c.parsed_reply,
            "goal_achieved": c.parsed_goal_achieved,
            "goal_type": c.parsed_goal_type,
            "extracted": c.parsed_extracted,
        },
        "duration_ms": c.duration_ms,
        "prompt_tokens": c.tokens_in,
        "completion_tokens": c.tokens_out,
        "error": c.error,
        "parse_failed": c.parse_failed,
    }


# Re-exported for the wrap-up wrapper (PR #9 will adopt this directly).
__all__ = [
    "PipelineOutcome",
    "run_pipeline",
    "PipelineConfig",
    "RoleSpec",
    "JudgeSpec",
    "PolishSpec",
]


def _ensure_dataclass_serializable(d: Any) -> Any:
    """Helper for tests."""
    return asdict(d) if hasattr(d, "__dataclass_fields__") else d
