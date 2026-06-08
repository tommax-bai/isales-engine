"""Unit tests for the composable barge-in rule tree (interruption_rules).

Change: engine-interruption-rule-tree.
"""

from __future__ import annotations

import pytest

from isales_engine.realtime.interruption_rules import (
    AndRule,
    DurationRule,
    KeywordRule,
    LengthRule,
    NoneRule,
    NotRule,
    OrRule,
    RegexRule,
    RuleInput,
    SplitByDelimiterRule,
    build_rule,
    default_rule,
)


def _ctx(text: str = "", elapsed_ms: int = 0) -> RuleInput:
    return RuleInput(text=text, elapsed_ms=elapsed_ms)


# ---- leaf rules ------------------------------------------------------------


def test_keyword_contains_substring_match_after_punct_trim() -> None:
    r = KeywordRule(values=("嗯", "好的"), match="contains")
    assert r.evaluate(_ctx("嗯嗯")).interrupt is True  # substring
    assert r.evaluate(_ctx("好的，")).interrupt is True  # trailing punct trimmed
    assert r.evaluate(_ctx("我想问个问题")).interrupt is False


def test_keyword_exact_is_whole_stripped_text() -> None:
    r = KeywordRule(values=("嗯", "好的"), match="exact")
    assert r.evaluate(_ctx("嗯")).interrupt is True
    assert r.evaluate(_ctx("  好的  ")).interrupt is True  # stripped
    # exact does NOT trim trailing punctuation (legacy whitelist parity) and is
    # not a substring match.
    assert r.evaluate(_ctx("嗯嗯")).interrupt is False
    assert r.evaluate(_ctx("好的，")).interrupt is False


def test_length_rule_rune_count() -> None:
    r = LengthRule(value=2)
    assert r.evaluate(_ctx("行")).interrupt is False
    assert r.evaluate(_ctx("等等")).interrupt is True
    assert r.evaluate(_ctx("  等  ")).interrupt is False  # stripped to 1 rune


def test_duration_rule_uses_elapsed_ms() -> None:
    r = DurationRule(value_ms=400)
    assert r.evaluate(_ctx("x", elapsed_ms=399)).interrupt is False
    assert r.evaluate(_ctx("x", elapsed_ms=400)).interrupt is True


def test_regex_rule_search() -> None:
    r = RegexRule(pattern="等.下")
    assert r.evaluate(_ctx("你等一下")).interrupt is True
    assert r.evaluate(_ctx("没问题")).interrupt is False


def test_split_by_delimiter_rule() -> None:
    r = SplitByDelimiterRule(delimiters=("，", "。"))
    assert r.evaluate(_ctx("你好，请问")).interrupt is True
    assert r.evaluate(_ctx("你好请问")).interrupt is False


def test_none_rule_always_false() -> None:
    assert NoneRule().evaluate(_ctx("anything")).interrupt is False


# ---- composites ------------------------------------------------------------


def test_and_short_circuits_on_first_false() -> None:
    r = AndRule((LengthRule(value=5), DurationRule(value_ms=0)))
    v = r.evaluate(_ctx("短", elapsed_ms=1000))
    assert v.interrupt is False
    assert v.reason == "length"  # the failing leaf


def test_and_all_true() -> None:
    r = AndRule((LengthRule(value=2), DurationRule(value_ms=100)))
    v = r.evaluate(_ctx("等等", elapsed_ms=200))
    assert v.interrupt is True
    assert v.reason == "and"


def test_or_short_circuits_on_first_true() -> None:
    r = OrRule((RegexRule(pattern="不可能命中"), LengthRule(value=2)))
    v = r.evaluate(_ctx("可以", elapsed_ms=0))
    assert v.interrupt is True
    assert v.reason == "length"


def test_or_all_false() -> None:
    r = OrRule((LengthRule(value=10), DurationRule(value_ms=9999)))
    v = r.evaluate(_ctx("短", elapsed_ms=0))
    assert v.interrupt is False
    assert v.reason == "or"


def test_not_inverts() -> None:
    inner = KeywordRule(values=("嗯",), match="exact")
    r = NotRule(inner)
    assert r.evaluate(_ctx("嗯")).interrupt is False
    assert r.evaluate(_ctx("我要打断")).interrupt is True


def test_nested_and_or_not() -> None:
    # AND( OR(regex, keyword), NOT(keyword 语气词), duration )
    r = AndRule(
        (
            OrRule((RegexRule(pattern="等一下"), KeywordRule(values=("打断",), match="contains"))),
            NotRule(KeywordRule(values=("嗯", "好的"), match="exact")),
            DurationRule(value_ms=100),
        )
    )
    assert r.evaluate(_ctx("等一下", elapsed_ms=200)).interrupt is True
    assert r.evaluate(_ctx("我要打断你", elapsed_ms=200)).interrupt is True
    # too soon → duration fails
    assert r.evaluate(_ctx("等一下", elapsed_ms=50)).interrupt is False
    # no OR match → ignored
    assert r.evaluate(_ctx("随便说点别的", elapsed_ms=200)).interrupt is False


# ---- factory ---------------------------------------------------------------


def test_build_rule_none_and_explicit_none() -> None:
    assert isinstance(build_rule(None), NoneRule)
    assert isinstance(build_rule({"type": "none"}), NoneRule)


def test_build_rule_recursive_tree() -> None:
    tree = build_rule(
        {
            "type": "and",
            "rules": [
                {"type": "not", "rule": {"type": "keyword", "values": ["嗯"], "match": "exact"}},
                {"type": "length", "value": 2},
                {"type": "duration", "value_ms": 400},
            ],
        }
    )
    assert tree.evaluate(_ctx("我要打断", elapsed_ms=500)).interrupt is True
    assert tree.evaluate(_ctx("嗯", elapsed_ms=500)).interrupt is False


def test_build_rule_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        build_rule({"type": "bogus"})
    with pytest.raises(ValueError):
        build_rule({"no_type": 1})


# ---- default tree (backward-compat) ---------------------------------------


def test_default_rule_reproduces_legacy_sequence() -> None:
    r = default_rule(whitelist=["嗯", "好的"], min_text_length=2, min_duration_ms=800)
    # whitelist hit → ignored
    assert r.evaluate(_ctx("嗯", elapsed_ms=2000)).interrupt is False
    # too short → ignored
    assert r.evaluate(_ctx("行", elapsed_ms=2000)).interrupt is False
    # too soon → ignored
    assert r.evaluate(_ctx("你看这个", elapsed_ms=500)).interrupt is False
    # all pass → triggered
    assert r.evaluate(_ctx("你看这个内容", elapsed_ms=900)).interrupt is True


def test_default_rule_empty_whitelist_omits_keyword_leaf() -> None:
    r = default_rule(whitelist=[], min_text_length=2, min_duration_ms=0)
    # no whitelist → only length + duration gate
    assert r.evaluate(_ctx("两字", elapsed_ms=0)).interrupt is True
    assert r.evaluate(_ctx("一", elapsed_ms=0)).interrupt is False
