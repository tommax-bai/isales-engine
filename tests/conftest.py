"""Shared pytest fixtures — real PG + Redis (skip if unreachable).

Mirrors the worker / scheduler test pattern so a single dev environment
serves all services.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from isales_common.models import Base
from isales_common.utils.redis import get_redis as common_redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

DEFAULT_DB_URL = "postgresql+asyncpg://bears@localhost:5432/isales_engine_test"
# db 2 — worker uses 0, scheduler uses 1.
DEFAULT_REDIS_URL = "redis://localhost:6379/2"

# Run the whole engine suite with transcript write-time validation in strict
# mode, so any test that drives a flow producing an out-of-contract transcript
# event fails loudly (engine-transcript-write-validation). setdefault lets an
# individual test opt out via monkeypatch to exercise the prod fail-soft path.
os.environ.setdefault("ISALES_ENGINE_STRICT_TRANSCRIPT", "1")


def _resolve_db_url() -> str:
    return (
        os.environ.get("ISALES_TEST_DATABASE_URL")
        or os.environ.get("ISALES_DATABASE_URL")
        or DEFAULT_DB_URL
    )


def _resolve_redis_url() -> str:
    return os.environ.get("ISALES_REDIS_URL", DEFAULT_REDIS_URL)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_resolve_db_url(), future=True, poolclass=NullPool)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        pytest.skip(f"PG not reachable: {exc}")
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_engine(engine: AsyncEngine) -> AsyncIterator[AsyncEngine]:
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "TRUNCATE campaign, role_config, lead, call_record, call_summary, "
            "callback_config, callback_log, holiday, prompt_version, "
            "pipeline_trace, device, sim_card, campaign_device, "
            "device_sim_binding "
            "RESTART IDENTITY CASCADE"
        )
    yield engine


@pytest_asyncio.fixture(loop_scope="session")
async def sessionmaker_(clean_engine: AsyncEngine) -> Any:
    return async_sessionmaker(clean_engine, expire_on_commit=False)


@pytest_asyncio.fixture(loop_scope="session")
async def redis_client() -> AsyncIterator[Any]:
    client = common_redis(_resolve_redis_url())
    try:
        await client.ping()
    except Exception as exc:
        pytest.skip(f"Redis not reachable: {exc}")
    keys = [
        "engine:dial",
        "engine:dlq",
        "engine:worker:call-ended",
        "isales:concurrency:active",
    ]
    for k in keys:
        await client.delete(k)
    yield client
    for k in keys:
        await client.delete(k)
    await client.aclose()
