"""Composable barge-in rule tree (port of voxen ``core/interruption/``).

Spec: interruption-detection § "可组合规则树打断判定" + "打断规则树配置与装配".
Change: engine-interruption-rule-tree.

A rule tree decides whether a user's ASR partial text counts as a barge-in. It is
built ONCE (``build_rule`` / ``default_rule``) when ``InterruptionConfig`` is
assembled and evaluated per ASR partial via :meth:`Rule.evaluate`.

Leaves test the partial text (and elapsed speech duration); composites combine
them with and/or/not. Each ``RuleVerdict`` carries a ``reason`` (provenance of the
deciding leaf) for logging / trace — mirrors voxen's ``InterruptionResult``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Trailing punctuation trimmed before a CONTAINS keyword match (voxen parity).
# EXACT mode does NOT trim — it reproduces the legacy ``text.strip() in whitelist``
# semantics byte-for-byte so the synthesized default tree matches today's behavior.
_TRAILING_PUNCT = "。，！？、；,.!?;"


@dataclass(frozen=True)
class RuleInput:
    """Per-partial evaluation context."""

    text: str
    elapsed_ms: int


@dataclass(frozen=True)
class RuleVerdict:
    """Outcome of evaluating a rule: interrupt-or-not + provenance reason."""

    interrupt: bool
    reason: str


@runtime_checkable
class Rule(Protocol):
    def evaluate(self, ctx: RuleInput) -> RuleVerdict: ...


# --- leaf rules ---------------------------------------------------------------


@dataclass
class KeywordRule:
    values: tuple[str, ...]
    match: str = "contains"  # "contains" | "exact"

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        if self.match == "exact":
            hit = ctx.text.strip() in self.values
        else:
            t = ctx.text.strip().rstrip(_TRAILING_PUNCT)
            hit = bool(t) and any(v in t for v in self.values)
        return RuleVerdict(hit, "keyword")


@dataclass
class LengthRule:
    value: int

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        return RuleVerdict(len(ctx.text.strip()) >= self.value, "length")


@dataclass
class DurationRule:
    value_ms: int

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        return RuleVerdict(ctx.elapsed_ms >= self.value_ms, "duration")


@dataclass
class RegexRule:
    pattern: str

    def __post_init__(self) -> None:
        self._re = re.compile(self.pattern)

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        return RuleVerdict(self._re.search(ctx.text) is not None, "regex")


@dataclass
class SplitByDelimiterRule:
    delimiters: tuple[str, ...]

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        return RuleVerdict(any(d in ctx.text for d in self.delimiters), "split_by_delimiter")


@dataclass
class NoneRule:
    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        return RuleVerdict(False, "none")


# --- composite rules ----------------------------------------------------------


@dataclass
class AndRule:
    rules: tuple[Rule, ...]

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        for r in self.rules:
            v = r.evaluate(ctx)
            if not v.interrupt:
                return RuleVerdict(False, v.reason)  # short-circuit: failing leaf
        return RuleVerdict(True, "and")


@dataclass
class OrRule:
    rules: tuple[Rule, ...]

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        for r in self.rules:
            v = r.evaluate(ctx)
            if v.interrupt:
                return RuleVerdict(True, v.reason)  # short-circuit: matching leaf
        return RuleVerdict(False, "or")


@dataclass
class NotRule:
    rule: Rule

    def evaluate(self, ctx: RuleInput) -> RuleVerdict:
        v = self.rule.evaluate(ctx)
        return RuleVerdict(not v.interrupt, f"not({v.reason})")


# --- factory ------------------------------------------------------------------


def build_rule(config: dict[str, Any] | None) -> Rule:
    """Compile a rule-tree dict (campaign.interruption_rules) into a Rule.

    ``None`` / ``{"type": "none"}`` → :class:`NoneRule` (interruption disabled).
    Raises ``ValueError`` on unknown ``type`` / malformed structure (装配期
    fail-fast; api validates before write, this is the runtime backstop).
    """
    if config is None:
        return NoneRule()
    if not isinstance(config, dict) or "type" not in config:
        raise ValueError(f"interruption rule must be a dict with a 'type': {config!r}")
    rtype = config["type"]
    if rtype == "none":
        return NoneRule()
    if rtype == "keyword":
        return KeywordRule(tuple(config["values"]), config.get("match", "contains"))
    if rtype == "length":
        return LengthRule(int(config["value"]))
    if rtype == "duration":
        return DurationRule(int(config["value_ms"]))
    if rtype == "regex":
        return RegexRule(str(config["pattern"]))
    if rtype == "split_by_delimiter":
        return SplitByDelimiterRule(tuple(config["delimiters"]))
    if rtype == "and":
        return AndRule(tuple(build_rule(r) for r in config["rules"]))
    if rtype == "or":
        return OrRule(tuple(build_rule(r) for r in config["rules"]))
    if rtype == "not":
        return NotRule(build_rule(config["rule"]))
    raise ValueError(f"unknown interruption rule type: {rtype!r}")


def default_rule(
    *, whitelist: list[str], min_text_length: int = 2, min_duration_ms: int
) -> Rule:
    """Synthesize the backward-compat default tree from legacy campaign columns.

    Equivalent to the historical ``evaluate_partial`` sequence (design D4):
    ``AND( NOT keyword(exact, whitelist), length(min_text_length), duration(min_duration_ms) )``.
    The ``NOT keyword`` leaf is omitted when ``whitelist`` is empty (an empty
    keyword set would never match — keeping it would be a harmless no-op but the
    omission keeps the tree minimal).
    """
    rules: list[Rule] = []
    if whitelist:
        rules.append(NotRule(KeywordRule(tuple(whitelist), "exact")))
    rules.append(LengthRule(min_text_length))
    rules.append(DurationRule(min_duration_ms))
    return AndRule(tuple(rules))
