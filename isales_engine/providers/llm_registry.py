"""Per-slot LLM client registry (per-role-llm-config-and-restructure-card).

Each engine-resident LLM slot (main / persona / referee / restructure) selects
its LLM by the role_config's **provider** (``ext_params.provider``, threaded onto
``_SlotSpec.provider``) + **model** (``role_config.model`` column). This registry
lazily builds and caches one client per ``(provider, model)`` for the duration of
a call, and falls back to a single global default client when a slot has no
provider configured or the provider's credentials are missing.

Built per-call alongside :class:`~isales_engine.run_loop.Providers`; call
:meth:`aclose_all` at call end to release every cached client's HTTP connection
pool (mirrors the per-call ``providers.llm.aclose()`` discipline).
"""

from __future__ import annotations

import contextlib
import logging

from isales_common.credentials import CredentialStore
from isales_common.providers.llm import LLMProvider

from isales_engine.providers.factory import build_llm

logger = logging.getLogger(__name__)


class LLMRegistry:
    """Resolve + cache LLM clients per (provider, model) with a single fallback."""

    def __init__(
        self,
        *,
        store: CredentialStore | None,
        default_provider: str,
        default_client: LLMProvider,
    ) -> None:
        self._store = store
        self._default_provider = default_provider
        self._default = default_client
        self._cache: dict[tuple[str, str | None], LLMProvider] = {}

    def resolve(self, provider: str | None, model: str | None) -> LLMProvider:
        """Return the LLM client for a slot's ``(provider, model)``.

        ``provider`` None/empty → the single global default client (the slot
        keeps the legacy "engine default model" behavior — this is the **one**
        permitted fallback layer, not a stacked just-in-case branch). On an
        unknown provider or missing credentials the registry logs a WARN and
        returns the default (fail-open: a misconfigured slot MUST NOT crash the
        call); the default is cached under the failing key so the failed build is
        not retried every turn (credentials are loaded once at startup and do not
        change at runtime — provider-credential spec).

        Removal trigger for the missing-provider fallback: drop it once
        ``provider`` becomes a required, UI-enforced field on every role_config.
        """
        if not provider:
            return self._default
        key = (provider, model)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            client = build_llm(provider, store=self._store, model=model)
        except (NotImplementedError, KeyError, ValueError) as exc:
            logger.warning(
                "llm_slot_provider_fallback provider=%s model=%s reason=%s "
                "falling_back_to=%s",
                provider,
                model,
                type(exc).__name__,
                self._default_provider,
            )
            self._cache[key] = self._default
            return self._default
        self._cache[key] = client
        return client

    def resolve_spec(self, spec: object) -> LLMProvider:
        """Resolve by a slot spec's ``provider`` + ``model`` attributes."""
        return self.resolve(
            getattr(spec, "provider", None), getattr(spec, "model", None)
        )

    async def aclose_all(self) -> None:
        """Release the default + every cached client's HTTP connections (per-call)."""
        seen: set[int] = set()
        for client in (self._default, *self._cache.values()):
            if id(client) in seen:
                continue
            seen.add(id(client))
            aclose = getattr(client, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()
