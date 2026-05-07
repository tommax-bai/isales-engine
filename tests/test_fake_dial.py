"""Smoke-test the fake_dial CLI's helper logic without invoking sys.argv."""

from __future__ import annotations

import argparse
from typing import Any

import pytest
from isales_common.models import Campaign

from scripts.fake_dial import _inject

pytestmark = [
    pytest.mark.usefixtures("clean_engine", "redis_client"),
    pytest.mark.asyncio(loop_scope="session"),
]


async def test_fake_dial_injects_message(
    sessionmaker_: Any, redis_client: Any
) -> None:
    # Seed a campaign so the (auto-created) lead has a valid FK.
    async with sessionmaker_() as db:
        c = Campaign(name="t", default_replies=["hi"])
        db.add(c)
        await db.commit()
        await db.refresh(c)

    db_url = "postgresql+asyncpg://bears@localhost:5432/isales_engine_test"
    redis_url = "redis://localhost:6379/2"

    args = argparse.Namespace(
        db_url=db_url,
        redis_url=redis_url,
        campaign_id=c.id,
        lead_id=None,
        phone_number="+8613800000000",
        caller_id="+8613900000000",
        device_id=1,
        queue="engine:dial",
        concurrency_key="isales:concurrency:active",
        no_incr=False,
    )

    await _inject(args)

    queue_len = await redis_client.llen("engine:dial")
    assert queue_len == 1
    counter = await redis_client.get("isales:concurrency:active")
    assert int(counter) == 1
