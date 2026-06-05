"""Re-export shim: provider error model + HTTP mapping now live in isales-common.

The exception classes and ``map_http_error`` / ``map_transport_error`` helpers
were moved to ``isales_common.providers._errors`` (campaign-greeting-tts-preview
§ 决策 1) so the shared volcengine TTS provider can be constructed from both
isales-engine and isales-api. This shim keeps the engine-local ASR / LLM
providers (``asr_volcengine``, ``llm_openai_compatible``) importing from
``isales_engine.providers._errors`` unchanged.

Removal trigger: drop this module once ``asr_volcengine`` and
``llm_openai_compatible`` import directly from ``isales_common.providers._errors``.
"""

from __future__ import annotations

from isales_common.providers._errors import (
    ProviderError,
    ProviderInvalidRequest,
    ProviderRateLimited,
    ProviderServerError,
    ProviderTimeout,
    as_provider_invalid_response,
    map_http_error,
    map_transport_error,
)

__all__ = [
    "ProviderError",
    "ProviderInvalidRequest",
    "ProviderRateLimited",
    "ProviderServerError",
    "ProviderTimeout",
    "as_provider_invalid_response",
    "map_http_error",
    "map_transport_error",
]
