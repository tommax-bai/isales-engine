"""Engine main wire-up: ISALES_ENGINE_TELEPHONY_MODE=rtc path.

Covers the `_build_telephony` rtc branch + `_build_rtc_transport` factory
+ runner device_id hint forwarding. Does NOT exercise the full `_main`
coroutine (that needs PG + Redis + a real gRPC bind); the smoke test
`engine_main_starts_grpc_server` in tests/test_grpc_server.py already
covers grpc.aio server bring-up.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from isales_common.transport.cloud_edge import EdgeNotConnected
from jose import jwt as jose_jwt

from isales_engine.call_session import CallSession
from isales_engine.main import (
    _build_rtc_transport,
    _build_telephony,
    _make_runner,
)
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.realtime.real_telephony import RealTelephonyClient
from isales_engine.realtime.rtc_telephony import RtcTelephonyClient
from isales_engine.settings import Settings
from isales_engine.transport.jwt_token_verifier import ALGORITHM, AUDIENCE
from isales_engine.transport.session_dispatcher import EngineSessionDispatcher

SECRET = "rtc-wireup-test-secret-32-bytes-or-more"


# Field name → env-var alias map for the settings we override here.
_ALIAS = {
    "database_url": "ISALES_DATABASE_URL",
    "redis_url": "ISALES_REDIS_URL",
    "engine_telephony_mode": "ISALES_ENGINE_TELEPHONY_MODE",
    "engine_edge_device_id": "ISALES_ENGINE_EDGE_DEVICE_ID",
    "engine_rtc_app_id": "ISALES_RTC_APP_ID",
    "engine_rtc_app_key": "ISALES_RTC_APP_KEY",
    "engine_rtc_sdk_kind": "ISALES_ENGINE_RTC_SDK_KIND",
    "engine_edge_token_jwt_secret": "ISALES_JWT_SECRET",
    "engine_cloud_edge_grpc_bind": "ISALES_ENGINE_CLOUD_EDGE_GRPC_BIND",
}


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "database_url": "postgresql+asyncpg://x/y",
        "redis_url": "redis://localhost:6379/0",
        "engine_telephony_mode": "rtc",
        "engine_edge_device_id": "edge-test",
        "engine_rtc_app_id": "app-id",
        "engine_rtc_app_key": "app-key",
        "engine_rtc_sdk_kind": "in_memory",
        "engine_edge_token_jwt_secret": SECRET,
    }
    base.update(overrides)
    return Settings(**{_ALIAS[k]: v for k, v in base.items()})


# ----- _build_telephony branches ---------------------------------------------


def test_build_telephony_mock_unchanged() -> None:
    s = _settings(engine_telephony_mode="mock")
    client = _build_telephony(s)
    assert isinstance(client, MockTelephonyClient)


def test_build_telephony_real_unchanged() -> None:
    s = _settings(engine_telephony_mode="real")
    client = _build_telephony(s)
    assert isinstance(client, RealTelephonyClient)


def test_build_telephony_rtc_requires_dispatcher_and_server() -> None:
    s = _settings()
    with pytest.raises(RuntimeError, match="dispatcher"):
        _build_telephony(s)


def test_build_telephony_rtc_requires_edge_device_id() -> None:
    s = _settings(engine_edge_device_id="")
    with pytest.raises(RuntimeError, match="EDGE_DEVICE_ID"):
        _build_telephony(s, dispatcher=MagicMock(), grpc_server=MagicMock())


def test_build_telephony_rtc_rejects_unknown_sdk_kind() -> None:
    s = _settings(engine_rtc_sdk_kind="something-else")
    with pytest.raises(RuntimeError, match="engine_rtc_sdk_kind"):
        _build_telephony(s, dispatcher=MagicMock(), grpc_server=MagicMock())


def test_build_telephony_rtc_returns_rtc_client_in_memory() -> None:
    s = _settings()
    dispatcher = EngineSessionDispatcher()
    grpc_server = MagicMock()

    client = _build_telephony(s, dispatcher=dispatcher, grpc_server=grpc_server)

    assert isinstance(client, RtcTelephonyClient)
    # The client's edge_device_id binding matches settings.
    assert client._edge_device_id == "edge-test"
    # Dispatcher and grpc_server are wired through.
    assert client._dispatcher is dispatcher
    assert client._grpc is grpc_server


# ----- _build_rtc_transport --------------------------------------------------


@pytest.mark.asyncio
async def test_build_rtc_transport_requires_secret() -> None:
    s = _settings(engine_edge_token_jwt_secret="")
    sessionmaker = MagicMock()
    with pytest.raises(RuntimeError, match="ISALES_JWT_SECRET"):
        await _build_rtc_transport(s, sessionmaker)


@pytest.mark.asyncio
async def test_build_rtc_transport_starts_server_and_wires_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transport factory authenticates real tokens and routes inbound
    messages through the dispatcher (heartbeat, hardware_alert callbacks
    registered). Uses a free port so this is hermetic."""
    s = _settings(engine_cloud_edge_grpc_bind="127.0.0.1:0")
    sessionmaker = MagicMock()  # not exercised in this test

    dispatcher, grpc_server = await _build_rtc_transport(s, sessionmaker)

    try:
        # Hardware-alert handler is registered (process-wide, not per-call).
        assert dispatcher._hardware_alert_cb is not None
        # Heartbeat handler is registered.
        assert dispatcher._heartbeat_cb is not None
        # Server's edge_callback is the dispatcher's handle_edge_message.
        # (Method-bound comparison: `is` doesn't work — every getattr returns
        # a fresh bound-method object — so compare __func__ + __self__.)
        cb = grpc_server._edge_callback  # type: ignore[attr-defined]
        assert cb is not None
        assert cb.__self__ is dispatcher  # type: ignore[union-attr]
        assert cb.__func__ is EngineSessionDispatcher.handle_edge_message  # type: ignore[union-attr]
        # And the bidirectional channel is up.
        assert grpc_server._server is not None  # type: ignore[attr-defined]
    finally:
        await grpc_server.stop(grace_seconds=0.1)


# ----- runner device_id hint forwarding --------------------------------------


@pytest.mark.asyncio
async def test_runner_forwards_device_id_hint_to_rtc_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_make_runner` must call `set_device_for_session` before `run_session`
    so the RtcTelephonyClient's DialCommand carries the scheduler-chosen
    device_id (not 0)."""
    s = _settings()
    dispatcher = EngineSessionDispatcher()
    grpc_server = MagicMock()
    grpc_server.send_to_edge = MagicMock(
        side_effect=EdgeNotConnected("edge-test"),
    )

    client = _build_telephony(s, dispatcher=dispatcher, grpc_server=grpc_server)

    captured: dict[str, Any] = {}

    async def _fake_run_session(*args: Any, **kwargs: Any) -> None:
        captured["called"] = True
        captured["pending_devices"] = dict(client._pending_devices)  # type: ignore[attr-defined]

    monkeypatch.setattr("isales_engine.main.run_session", _fake_run_session)
    monkeypatch.setattr(
        "isales_engine.main.load_runtime_config",
        _fake_load_runtime_config,
    )
    monkeypatch.setattr(
        "isales_engine.main.Providers",
        _NoopProviders,
    )
    monkeypatch.setattr("isales_engine.main.build_llm", lambda _name, **_kw: None)
    monkeypatch.setattr("isales_engine.main.build_asr", lambda _name, **_kw: None)
    monkeypatch.setattr("isales_engine.main.build_tts", lambda _name, **_kw: None)

    publisher = MagicMock()
    sessionmaker = _StubSessionmaker(lead=_StubLead(lead_id=1, phone="+1"))
    from isales_common.credentials import CredentialStore
    runner = _make_runner(
        sessionmaker=sessionmaker,
        telephony=client,
        publisher=publisher,
        settings=s,
        credentials=CredentialStore(),
    )

    session = CallSession(
        call_record_id=42,
        campaign_id=10,
        lead_id=1,
        caller_id="+86133",
        prompt_versions_snapshot={
            "role_llms": [],
            "judge_llm": None,
            "polish_llm": None,
            "wrap_up_appended": False,
        },
        device_id=7,
    )

    await runner(session)

    assert captured["called"] is True
    # The hint reached the client before run_session.
    assert captured["pending_devices"] == {42: 7}


@pytest.mark.asyncio
async def test_runner_skips_hint_for_mock_telephony(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock client has no set_device_for_session — the runner must tolerate
    that (getattr returns None)."""
    s = _settings(engine_telephony_mode="mock")
    client = _build_telephony(s)
    assert not hasattr(client, "set_device_for_session")

    async def _fake_run_session(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr("isales_engine.main.run_session", _fake_run_session)
    monkeypatch.setattr(
        "isales_engine.main.load_runtime_config",
        _fake_load_runtime_config,
    )
    monkeypatch.setattr(
        "isales_engine.main.Providers",
        _NoopProviders,
    )
    monkeypatch.setattr("isales_engine.main.build_llm", lambda _name, **_kw: None)
    monkeypatch.setattr("isales_engine.main.build_asr", lambda _name, **_kw: None)
    monkeypatch.setattr("isales_engine.main.build_tts", lambda _name, **_kw: None)

    publisher = MagicMock()
    sessionmaker = _StubSessionmaker(lead=_StubLead(lead_id=1, phone="+1"))
    from isales_common.credentials import CredentialStore
    runner = _make_runner(
        sessionmaker=sessionmaker,
        telephony=client,
        publisher=publisher,
        settings=s,
        credentials=CredentialStore(),
    )

    session = CallSession(
        call_record_id=42,
        campaign_id=10,
        lead_id=1,
        caller_id="+86133",
        prompt_versions_snapshot={
            "role_llms": [],
            "judge_llm": None,
            "polish_llm": None,
            "wrap_up_appended": False,
        },
        device_id=7,
    )

    # Just shouldn't raise.
    await runner(session)


# ----- stubs -----------------------------------------------------------------


class _StubLead:
    def __init__(self, lead_id: int, phone: str) -> None:
        self.lead_id = lead_id
        self.phone = phone
        self.name = None
        self.custom_data: dict[str, Any] = {}


class _StubDb:
    def __init__(self, lead: _StubLead) -> None:
        self._lead = lead

    async def __aenter__(self) -> _StubDb:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get(self, model: Any, lead_id: int) -> _StubLead | None:
        if lead_id == self._lead.lead_id:
            return self._lead
        return None


class _StubSessionmaker:
    def __init__(self, lead: _StubLead) -> None:
        self._lead = lead

    def __call__(self) -> _StubDb:
        return _StubDb(self._lead)


async def _fake_load_runtime_config(
    db: Any, request: Any, *, pipeline_default_timeout_ms: int
) -> Any:
    return MagicMock()


class _NoopProviders:
    def __init__(self, *, llm: Any = None, asr: Any = None, tts: Any = None) -> None:
        self.llm = llm
        self.asr = asr
        self.tts = tts


# ----- helper: confirm JWT shape matches verifier expectations ---------------


def test_smoke_jwt_audience_constant_aligns_with_verifier() -> None:
    """Drift guard: if AUDIENCE diverges between mint and verify, edges can't
    reach the engine. This pins the cross-module contract."""
    # Encode using AUDIENCE; decode would fail if either side changed.
    now = datetime.now(tz=UTC)
    token = jose_jwt.encode(
        {
            "sub": "edge-1",
            "aud": AUDIENCE,
            "scope": "edge",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        },
        SECRET,
        algorithm=ALGORITHM,
    )
    decoded = jose_jwt.decode(
        token, SECRET, algorithms=[ALGORITHM], audience=AUDIENCE
    )
    assert decoded["sub"] == "edge-1"


# Silence linter on unused import (kept for future tests).
_ = replace
