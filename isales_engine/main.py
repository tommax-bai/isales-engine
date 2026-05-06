"""isales-engine entrypoint.

PR #1 wires the lifespan skeleton: settings → engine/redis clients → signal
handler → ``stop_event.wait()``. Subsequent PRs add the actual long-running
tasks (dial consumer / event consumer / session manager) to ``_main``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from isales_engine.db import get_engine, get_sessionmaker
from isales_engine.redis_client import get_redis
from isales_engine.settings import load_settings

logger = logging.getLogger(__name__)


async def _main() -> None:
    settings = load_settings()
    engine = get_engine(settings.database_url)
    sessionmaker = get_sessionmaker(engine)
    redis = get_redis(settings.redis_url)

    # sessionmaker is unused at the skeleton stage; it gets wired into
    # dial_consumer / transcript_recorder in PR #3 / PR #11.
    del sessionmaker

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        logger.info("shutdown_signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    logger.info("isales_engine_started")
    try:
        await stop_event.wait()
    finally:
        await redis.close()
        await engine.dispose()
        logger.info("isales_engine_stopped")


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_main())


if __name__ == "__main__":
    run()
