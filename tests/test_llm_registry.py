"""LLMRegistry — per-slot (provider, model) resolution + single fallback.

per-role-llm-config-and-restructure-card. Verifies each slot selects its own
provider+model, no-provider falls back to the global default, a missing
credential / unknown provider fails open to the default (with a WARN), and
clients are cached per (provider, model).
"""

from __future__ import annotations

import logging

import pytest

from isales_engine.providers import llm_registry as reg_mod
from isales_engine.providers.llm_registry import LLMRegistry


class _FakeClient:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _make(monkeypatch, *, bad: frozenset[str] = frozenset()):
    built: list[tuple[str, str | None]] = []

    def fake_build_llm(provider, *, store=None, model=None):
        built.append((provider, model))
        if provider in bad:
            raise NotImplementedError(f"no credential for {provider!r}")
        return _FakeClient(f"{provider}:{model}")

    monkeypatch.setattr(reg_mod, "build_llm", fake_build_llm)
    default = _FakeClient("DEFAULT")
    registry = LLMRegistry(
        store=None, default_provider="volcengine", default_client=default
    )
    return registry, default, built


def test_no_provider_returns_global_default(monkeypatch):
    r, default, built = _make(monkeypatch)
    assert r.resolve(None, "anything") is default
    assert r.resolve("", "anything") is default
    assert built == []  # never builds a client for an empty provider


def test_slot_uses_its_provider_and_model(monkeypatch):
    r, default, built = _make(monkeypatch)
    client = r.resolve("dashscope", "qwen-plus")
    assert isinstance(client, _FakeClient)
    assert client.tag == "dashscope:qwen-plus"
    assert client is not default
    assert built == [("dashscope", "qwen-plus")]


def test_distinct_provider_model_get_distinct_clients(monkeypatch):
    r, _default, built = _make(monkeypatch)
    a = r.resolve("volcengine", "doubao-pro")
    b = r.resolve("dashscope", "qwen-plus")
    assert a is not b
    assert built == [("volcengine", "doubao-pro"), ("dashscope", "qwen-plus")]


def test_same_provider_model_is_cached(monkeypatch):
    r, _default, built = _make(monkeypatch)
    a = r.resolve("openai", "gpt-4o")
    b = r.resolve("openai", "gpt-4o")
    assert a is b
    assert built == [("openai", "gpt-4o")]  # built exactly once


def test_missing_credential_falls_back_to_default_and_warns(monkeypatch, caplog):
    r, default, built = _make(monkeypatch, bad=frozenset({"openai"}))
    with caplog.at_level(logging.WARNING):
        client = r.resolve("openai", "gpt-4o")
    assert client is default
    assert any(
        "llm_slot_provider_fallback" in rec.getMessage() for rec in caplog.records
    )
    # The failed build is cached as the default → not retried every turn.
    again = r.resolve("openai", "gpt-4o")
    assert again is default
    assert built == [("openai", "gpt-4o")]  # only one build attempt


def test_resolve_spec_reads_provider_and_model_attrs(monkeypatch):
    r, _default, _built = _make(monkeypatch)

    class _Spec:
        provider = "dashscope"
        model = "qwen-max"

    assert r.resolve_spec(_Spec()).tag == "dashscope:qwen-max"


def test_resolve_spec_without_provider_uses_default(monkeypatch):
    r, default, _built = _make(monkeypatch)

    class _Spec:
        provider = None
        model = "mock"

    assert r.resolve_spec(_Spec()) is default


async def test_aclose_all_closes_default_and_every_cached_client(monkeypatch):
    r, default, _built = _make(monkeypatch)
    c1 = r.resolve("openai", "gpt-4o")
    c2 = r.resolve("dashscope", "qwen-plus")
    await r.aclose_all()
    assert default.closed
    assert c1.closed
    assert c2.closed
