"""Engine-side factory routing for the volcengine TTS provider.

The provider implementation + its SSE/error/reuse tests moved to isales-common
(campaign-greeting-tts-preview § 决策 1); see
isales-common/tests/test_tts_volcengine.py. What remains engine-specific is
that ``isales_engine.providers.factory.build_tts`` routes "volcengine" to the
shared constructor.
"""

from __future__ import annotations

import pytest

from isales_common.providers._errors import ProviderInvalidRequest
from isales_common.providers.tts_volcengine import VolcengineTTSProvider


def test_factory_volcengine_requires_credentials() -> None:
    """缺凭据字段 = ProviderInvalidRequest (shared build_volcengine_tts;
    callers surface 4xx, not 500). 历史上 engine 抛 NotImplementedError,
    下沉 common 后统一为 ProviderInvalidRequest。"""
    from isales_common.credentials import CredentialStore

    from isales_engine.providers.factory import build_tts

    empty_store = CredentialStore()
    with pytest.raises(ProviderInvalidRequest, match="app_key"):
        build_tts("volcengine", store=empty_store)


def test_factory_volcengine_with_credentials() -> None:
    from isales_common.credentials import CredentialStore

    from isales_engine.providers.factory import build_tts

    store = CredentialStore({"volcengine": {"app_key": "k", "app_token": "t"}})
    provider = build_tts("volcengine", store=store)
    assert isinstance(provider, VolcengineTTSProvider)
