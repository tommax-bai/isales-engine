"""isales-engine entrypoint.

Wires the four long-running coroutines:

* ``dial_consumer.dial_loop`` — BLPOPs ``engine:dial`` and spawns per-call
  ``run_session`` tasks (PR #11 default runner).
* ``event_consumer.subscribe_loop`` — PSUBSCRIBEs ``engine:control:campaign:*``
  and dispatches ``ManualHangup`` / ``TransferCommand`` to the live session.
* ``session_manager.cancel_all`` — invoked in the SIGTERM finally block to
  let in-flight calls finalize cleanly (PR #13 deepens this further).

Real LLM/ASR/TTS providers and the modem-controller IPC arrive in stages 5
and 6; the factory still wires the mock variants by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from isales_engine.call_session import CallSession
from isales_engine.db import get_engine, get_sessionmaker
from isales_engine.dial_consumer import dial_loop
from isales_engine.event_consumer import subscribe_loop
from isales_engine.event_publisher import EventPublisher
from isales_engine.providers.factory import build_asr, build_llm, build_tts
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.realtime.real_telephony import RealTelephonyClient
from isales_engine.realtime.telephony_client import TelephonyClient
from isales_engine.redis_client import get_redis
from isales_engine.run_loop import (
    Providers,
    request_manual_hangup,
    run_session,
)
from isales_engine.runtime_config import load_runtime_config
from isales_engine.session_manager import SessionManager
from isales_engine.settings import Settings, load_settings

logger = logging.getLogger(__name__)


def _build_telephony(settings: Settings) -> TelephonyClient:
    if settings.engine_telephony_mode == "mock":
        return MockTelephonyClient(
            connect_delay_ms=settings.engine_mock_connect_delay_ms
        )
    if settings.engine_telephony_mode == "real":
        return RealTelephonyClient(
            settings.engine_telephony_socket_path,
            dial_timeout_s=settings.engine_telephony_dial_timeout_s,
        )
    raise NotImplementedError(
        f"telephony_mode {settings.engine_telephony_mode!r} not wired"
    )


def _make_runner(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    telephony: TelephonyClient,
    publisher: EventPublisher,
    settings: Settings,
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
                device_id=0,
            )
            runtime = await load_runtime_config(
                db,
                request,
                pipeline_default_timeout_ms=settings.engine_pipeline_default_timeout_ms,
            )

        providers = Providers(
            llm=build_llm(settings.engine_llm_provider),
            asr=build_asr(settings.engine_asr_provider),
            tts=build_tts(settings.engine_tts_provider),
        )

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

    return _run


async def _main() -> None:
    settings = load_settings()
    db_engine = get_engine(settings.database_url)
    sessionmaker = get_sessionmaker(db_engine)
    redis = get_redis(settings.redis_url)

    session_manager = SessionManager()
    publisher = EventPublisher(redis)
    telephony = _build_telephony(settings)

    runner = _make_runner(
        sessionmaker=sessionmaker,
        telephony=telephony,
        publisher=publisher,
        settings=settings,
    )

    async def _on_manual_hangup(call_record_id: int, _operator: str | None) -> None:
        sess = session_manager.get(call_record_id)
        if sess is None:
            return
        request_manual_hangup(sess)

    async def _on_transfer(call_record_id: int, _agent_id: int) -> None:
        # Stage 4: route via manual hangup; richer routing arrives in stage 6.
        sess = session_manager.get(call_record_id)
        if sess is None:
            return
        request_manual_hangup(sess)

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
