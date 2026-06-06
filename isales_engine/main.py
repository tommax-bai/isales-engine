"""isales-engine entrypoint.

Wires the four long-running coroutines:

* ``dial_consumer.dial_loop`` — BLPOPs ``engine:dial`` and spawns per-call
  ``run_session`` tasks (PR #11 default runner).
* ``event_consumer.subscribe_loop`` — PSUBSCRIBEs ``engine:control:campaign:*``
  and dispatches ``ManualHangup`` / ``TransferCommand`` to the live session.
* ``session_manager.cancel_all`` — invoked in the SIGTERM finally block to
  let in-flight calls finalize cleanly (PR #13 deepens this further).

For ``ISALES_ENGINE_TELEPHONY_MODE=rtc`` (arch-cloud-edge-split A2) the
cloud-edge gRPC server is also started here, with its inbound callback
routed through :class:`EngineSessionDispatcher` to per-call sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable

from isales_common.credentials import CredentialStore
from isales_common.transport.cloud_edge import CloudEdgeServer
from isales_common.utils.crypto import CryptoConfigError, CryptoError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from isales_engine.call_session import CallSession
from isales_engine.db import get_engine, get_sessionmaker
from isales_engine.dial_consumer import dial_loop
from isales_engine.event_consumer import subscribe_loop
from isales_engine.event_publisher import EventPublisher
from isales_engine.events import ManualHangupRequested, TransferRequested
from isales_engine.providers.factory import build_asr, build_llm, build_tts
from isales_engine.providers.tts_cache import CachingTTSProvider, TtsCacheStore
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.realtime.real_telephony import RealTelephonyClient
from isales_engine.realtime.rtc_telephony import RtcTelephonyClient
from isales_engine.realtime.telephony_client import TelephonyClient
from isales_engine.redis_client import get_redis
from isales_engine.run_loop import (
    Providers,
    run_session,
)
from isales_engine.runtime_config import load_runtime_config
from isales_engine.session_manager import SessionManager
from isales_engine.settings import Settings, load_settings
from isales_engine.transport.dingrtc import (
    DingRtcSession,
    InMemoryDingRtcChannel,
)
from isales_engine.transport.grpc_server import CloudEdgeGrpcServer
from isales_engine.transport.hardware_alert_handler import log_hardware_alert
from isales_engine.transport.heartbeat_handler import make_heartbeat_handler
from isales_engine.transport.jwt_token_verifier import JwtTokenVerifier
from isales_engine.transport.rtc_token import RtcTokenIssuer
from isales_engine.transport.session_dispatcher import EngineSessionDispatcher

logger = logging.getLogger(__name__)


def _build_telephony(
    settings: Settings,
    *,
    dispatcher: EngineSessionDispatcher | None = None,
    grpc_server: CloudEdgeServer | None = None,
) -> TelephonyClient:
    mode = settings.engine_telephony_mode
    if mode == "mock":
        return MockTelephonyClient(
            connect_delay_ms=settings.engine_mock_connect_delay_ms
        )
    if mode == "real":
        return RealTelephonyClient(
            settings.engine_telephony_socket_path,
            dial_timeout_s=settings.engine_telephony_dial_timeout_s,
        )
    if mode == "rtc":
        if dispatcher is None or grpc_server is None:
            raise RuntimeError(
                "rtc telephony requires dispatcher + grpc_server",
            )
        if not settings.engine_edge_device_id:
            raise RuntimeError(
                "ISALES_ENGINE_EDGE_DEVICE_ID must be set for rtc mode",
            )
        issuer = RtcTokenIssuer(
            app_id=settings.engine_rtc_app_id,
            app_key=settings.engine_rtc_app_key,
        )
        sdk_kind = settings.engine_rtc_sdk_kind
        if sdk_kind == "vendor":
            def rtc_factory() -> DingRtcSession:
                return DingRtcSession.production(
                    app_id=settings.engine_rtc_app_id,
                )
        elif sdk_kind == "in_memory":
            def rtc_factory() -> DingRtcSession:
                return DingRtcSession(
                    channel=InMemoryDingRtcChannel(),
                    app_id=settings.engine_rtc_app_id,
                )
        else:
            raise RuntimeError(
                f"unknown engine_rtc_sdk_kind: {sdk_kind!r}",
            )
        return RtcTelephonyClient(
            edge_device_id=settings.engine_edge_device_id,
            grpc_server=grpc_server,
            dispatcher=dispatcher,
            token_issuer=issuer,
            rtc_session_factory=rtc_factory,
        )
    raise NotImplementedError(f"telephony_mode {mode!r} not wired")


async def _load_credentials(
    sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> CredentialStore:
    """Load provider credentials from DB at startup.

    Failure handling per provider-credential spec § "装载失败时硬失败":
    - credentials_required=True (default): exit (let systemd restart-loop
      reveal the problem).
    - credentials_required=False (dev/CI): warn + return empty store so
      every non-mock build_* path will NotImplementedError when invoked;
      pipeline can still run with engine_*_provider=mock.
    """
    try:
        async with sessionmaker() as db:
            store = await CredentialStore.from_db(db)
        logger.info(
            "credentials_loaded count=%d providers=%s",
            store.row_count(),
            sorted(store.providers()),
        )
        return store
    except (CryptoConfigError, CryptoError) as exc:
        if settings.credentials_required:
            logger.error("cred_load_failed reason=%s", exc)
            raise SystemExit(
                f"cred_load_failed: {exc}. See RUNBOOK § '凭据轮换 / 主密钥'."
            ) from exc
        logger.warning(
            "cred_load_failed_falling_back_to_mock reason=%s (CREDENTIALS_REQUIRED=false)",
            exc,
        )
        return CredentialStore()
    except Exception as exc:  # pragma: no cover - DB unreachable
        if settings.credentials_required:
            logger.error("cred_load_failed reason=db_unreachable: %s", exc)
            raise SystemExit(f"cred_load_failed: {exc}") from exc
        logger.warning(
            "cred_load_failed_falling_back_to_mock reason=%s (CREDENTIALS_REQUIRED=false)",
            exc,
        )
        return CredentialStore()


def _make_runner(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    telephony: TelephonyClient,
    publisher: EventPublisher,
    settings: Settings,
    credentials: CredentialStore,
    tts_cache: TtsCacheStore,
) -> Callable[[CallSession], Awaitable[None]]:
    """Return the ``runner`` callable consumed by ``dial_consumer.dial_loop``."""

    async def _run(session: CallSession) -> None:
        async with sessionmaker() as db:
            from isales_common.models import Lead

            lead = await db.get(Lead, session.lead_id)
            phone = lead.phone if lead else "+0000000000"
            from isales_common.schemas.messages.dial import (
                DialLeadInfo,
                DialRequest,
                PromptVersionsSnapshot,
            )

            # Reconstruct a DialRequest-shaped object for runtime_config so we
            # don't need to thread DialRequest through the session itself.
            request = DialRequest(
                lead=DialLeadInfo(
                    lead_id=session.lead_id,
                    campaign_id=session.campaign_id,
                    phone=phone,
                    name=lead.name if lead else None,
                    custom_data=dict(lead.custom_data) if lead else {},
                ),
                history=[],
                prompt_versions=PromptVersionsSnapshot(
                    **session.prompt_versions_snapshot
                ),
                caller_id=session.caller_id,
                device_id=session.device_id,
            )
            runtime = await load_runtime_config(
                db,
                request,
                pipeline_default_timeout_ms=settings.engine_pipeline_default_timeout_ms,
            )

        providers = Providers(
            llm=build_llm(settings.engine_llm_provider, store=credentials),
            asr=build_asr(
                settings.engine_asr_provider,
                store=credentials,
                partial_stable_s=runtime.asr_partial_stable_s,
            ),
            tts=CachingTTSProvider(
                build_tts(settings.engine_tts_provider, store=credentials),
                tts_cache,
            ),
        )

        # The DialCommand on the edge carries the scheduler-selected
        # device_id; the modem-controller picks the right serial port from
        # it. Real + Rtc implementations both expose this hint surface;
        # Mock doesn't need it.
        hint = getattr(telephony, "set_device_for_session", None)
        if callable(hint):
            hint(session.call_record_id, session.device_id)

        try:
            await run_session(
                session,
                phone=phone,
                config=runtime,
                telephony=telephony,
                providers=providers,
                publisher=publisher,
                pipeline_timeout_ms=settings.engine_pipeline_default_timeout_ms,
                token_budget_per_call=settings.engine_token_budget_per_call,
            )
        finally:
            # Providers are per-call; release the TTS + LLM providers'
            # persistent HTTP clients so their keep-alive connections aren't
            # leaked (pipeline-latency-tail § C — TTS + LLM connection reuse).
            for provider in (providers.tts, providers.llm):
                aclose = getattr(provider, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await aclose()

    return _run


async def _build_rtc_transport(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> tuple[EngineSessionDispatcher, CloudEdgeGrpcServer]:
    """Build + start the cloud-edge gRPC server for rtc mode.

    Returns the dispatcher + server pair so the caller can pass the
    dispatcher into :func:`_build_telephony` and the server into the
    shutdown path.
    """
    if not settings.engine_edge_token_jwt_secret:
        raise RuntimeError(
            "ISALES_JWT_SECRET must be set for rtc mode "
            "(shared with isales-api edge-token mint)",
        )
    verifier = JwtTokenVerifier(secret=settings.engine_edge_token_jwt_secret)
    grpc_server = CloudEdgeGrpcServer(token_verifier=verifier)
    dispatcher = EngineSessionDispatcher()
    dispatcher.on_hardware_alert(log_hardware_alert)
    dispatcher.on_heartbeat(make_heartbeat_handler(sessionmaker))
    grpc_server.on_edge_message(dispatcher.handle_edge_message)
    await grpc_server.start(settings.engine_cloud_edge_grpc_bind)
    return dispatcher, grpc_server


async def _main() -> None:
    settings = load_settings()
    db_engine = get_engine(settings.database_url)
    sessionmaker = get_sessionmaker(db_engine)
    redis = get_redis(settings.redis_url)

    session_manager = SessionManager()
    publisher = EventPublisher(redis)

    dispatcher: EngineSessionDispatcher | None = None
    grpc_server: CloudEdgeGrpcServer | None = None
    if settings.engine_telephony_mode == "rtc":
        dispatcher, grpc_server = await _build_rtc_transport(settings, sessionmaker)

    telephony = _build_telephony(
        settings,
        dispatcher=dispatcher,
        grpc_server=grpc_server,
    )

    credentials = await _load_credentials(sessionmaker, settings)

    # Process-level fixed-phrase TTS cache shared across all calls
    # (tts-cache-and-gated-filler § A): greeting / silence / transfer / filler
    # phrases synthesize once per process, then replay zero-synth.
    tts_cache = TtsCacheStore()

    runner = _make_runner(
        sessionmaker=sessionmaker,
        telephony=telephony,
        publisher=publisher,
        settings=settings,
        credentials=credentials,
        tts_cache=tts_cache,
    )

    async def _on_manual_hangup(call_record_id: int, operator: str | None) -> None:
        sess = session_manager.get(call_record_id)
        if sess is None or sess.bus is None:
            return
        # run_session's control bridge cancels the run-loop 'main' task with
        # MANUAL_HANGUP (engine-eventbus-foundation). sess.bus is None only in
        # the brief window before run_session starts the bus — graceful skip.
        sess.bus.post(ManualHangupRequested(operator=operator))

    async def _on_transfer(call_record_id: int, agent_id: int) -> None:
        # Stage 4: transfer routes through the same hangup path (richer routing
        # arrives in a later change). The control bridge in run_session handles
        # the TransferRequested event byte-identically to manual hangup.
        sess = session_manager.get(call_record_id)
        if sess is None or sess.bus is None:
            return
        sess.bus.post(TransferRequested(agent_id=str(agent_id)))

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        logger.info("shutdown_signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    dial_task = asyncio.create_task(
        dial_loop(
            redis=redis,
            sessionmaker=sessionmaker,
            settings=settings,
            session_manager=session_manager,
            runner=runner,
        ),
        name="dial_loop",
    )
    control_task = asyncio.create_task(
        subscribe_loop(
            redis=redis,
            session_manager=session_manager,
            on_manual_hangup=_on_manual_hangup,
            on_transfer_command=_on_transfer,
        ),
        name="control_loop",
    )

    logger.info("isales_engine_started")
    try:
        await stop_event.wait()
    finally:
        logger.info("graceful_shutdown_begin active=%s", session_manager.active_count())
        for task in (dial_task, control_task):
            task.cancel()
        for task in (dial_task, control_task):
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await task

        await session_manager.cancel_all(
            timeout_s=float(settings.engine_graceful_shutdown_timeout_s)
        )
        if grpc_server is not None:
            await grpc_server.stop(grace_seconds=5.0)
        await publisher.drain()
        await redis.close()
        await db_engine.dispose()
        logger.info("isales_engine_stopped")


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_main())


if __name__ == "__main__":
    run()
