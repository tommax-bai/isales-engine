"""Tests for transfer.manager + wrapup.manager."""

from __future__ import annotations

from isales_engine.providers.llm_mock import KeywordDrivenMockLLM
from isales_engine.transfer.manager import (
    TransferConfig,
    evaluate_transfer,
)
from isales_engine.wrapup.manager import WrapUpConfig, evaluate_wrap_up

# ---- transfer trigger -----------------------------------------------------


def _config(
    *,
    keyword_enabled: bool = False,
    keywords: tuple[str, ...] = (),
    intent_enabled: bool = False,
    intent_threshold: float | None = None,
    round_enabled: bool = False,
    round_threshold: int | None = None,
    llm_enabled: bool = False,
) -> TransferConfig:
    return TransferConfig(
        keyword_enabled=keyword_enabled,
        keywords=keywords,
        intent_enabled=intent_enabled,
        intent_threshold=intent_threshold,
        intent_system_prompt="判定用户是否要转人工",
        round_enabled=round_enabled,
        round_threshold=round_threshold,
        llm_enabled=llm_enabled,
        llm_system_prompt="独立判定是否转人工",
        phrases=("您稍等，专员稍后联系您。", "好的，已经登记。"),
    )


async def test_transfer_no_triggers_enabled_returns_false() -> None:
    decision = await evaluate_transfer(
        user_text="您好",
        turn_count=1,
        goal_achieved=False,
        config=_config(),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is False


async def test_transfer_keyword_match() -> None:
    decision = await evaluate_transfer(
        user_text="我要转人工",
        turn_count=1,
        goal_achieved=False,
        config=_config(keyword_enabled=True, keywords=("转人工", "人工服务")),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is True
    assert decision.trigger_type == "keyword"
    assert "转人工" in decision.trigger_detail
    assert decision.phrase  # one of the configured phrases


async def test_transfer_round_threshold() -> None:
    decision = await evaluate_transfer(
        user_text="嗯",
        turn_count=10,
        goal_achieved=False,
        config=_config(round_enabled=True, round_threshold=8),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is True
    assert decision.trigger_type == "round"


async def test_transfer_round_threshold_blocked_by_goal_achieved() -> None:
    decision = await evaluate_transfer(
        user_text="嗯",
        turn_count=10,
        goal_achieved=True,
        config=_config(round_enabled=True, round_threshold=8),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is False


async def test_transfer_intent_via_llm() -> None:
    decision = await evaluate_transfer(
        user_text="我要转人工",  # KeywordDrivenMock returns probability=0.95 for this
        turn_count=1,
        goal_achieved=False,
        config=_config(intent_enabled=True, intent_threshold=0.8),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is True
    assert decision.trigger_type == "intent"


async def test_transfer_intent_below_threshold() -> None:
    decision = await evaluate_transfer(
        user_text="随便聊聊",  # mock probability=0.1
        turn_count=1,
        goal_achieved=False,
        config=_config(intent_enabled=True, intent_threshold=0.8),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is False


async def test_transfer_independent_llm() -> None:
    decision = await evaluate_transfer(
        user_text="我要投诉",  # mock returns transfer=true
        turn_count=1,
        goal_achieved=False,
        config=_config(llm_enabled=True),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is True
    assert decision.trigger_type == "llm"


async def test_transfer_or_keyword_first_match_wins() -> None:
    """When all four are enabled and a keyword matches first, we don't even
    reach the LLM-bearing detectors (keeps cost / latency low)."""

    decision = await evaluate_transfer(
        user_text="我要转人工",
        turn_count=10,
        goal_achieved=False,
        config=_config(
            keyword_enabled=True,
            keywords=("转人工",),
            round_enabled=True,
            round_threshold=8,
            intent_enabled=True,
            intent_threshold=0.5,
            llm_enabled=True,
        ),
        llm=KeywordDrivenMockLLM(),
    )
    assert decision.triggered is True
    # Keyword check is first → keyword wins.
    assert decision.trigger_type == "keyword"


# ---- wrapup ----------------------------------------------------------------


def _wcfg(
    *,
    max_rounds: int = 2,
    max_seconds: int = 15,
    closing: tuple[str, ...] = ("好的再见。",),
) -> WrapUpConfig:
    return WrapUpConfig(
        max_rounds=max_rounds, max_seconds=max_seconds, closing_phrases=closing
    )


def test_wrapup_proceeds_when_under_thresholds() -> None:
    d = evaluate_wrap_up(
        rounds_so_far=0,
        started_at_monotonic=100.0,
        config=_wcfg(),
        now_monotonic=105.0,
    )
    assert d.proceed is True


def test_wrapup_rounds_exhausted_returns_closing_phrase() -> None:
    d = evaluate_wrap_up(
        rounds_so_far=2, started_at_monotonic=100.0, config=_wcfg(), now_monotonic=101.0
    )
    assert d.proceed is False
    assert d.reason == "rounds_exhausted"
    assert d.closing_phrase == "好的再见。"


def test_wrapup_seconds_exhausted() -> None:
    d = evaluate_wrap_up(
        rounds_so_far=1,
        started_at_monotonic=100.0,
        config=_wcfg(max_seconds=15),
        now_monotonic=120.0,
    )
    assert d.proceed is False
    assert d.reason == "seconds_exhausted"


def test_wrapup_no_started_at_means_only_rounds_apply() -> None:
    d = evaluate_wrap_up(
        rounds_so_far=0, started_at_monotonic=None, config=_wcfg(), now_monotonic=99999.0
    )
    assert d.proceed is True


def test_wrapup_empty_phrases_falls_back() -> None:
    d = evaluate_wrap_up(
        rounds_so_far=99,
        started_at_monotonic=None,
        config=_wcfg(closing=()),
    )
    assert d.proceed is False
    assert d.closing_phrase == "好的，再见。"


