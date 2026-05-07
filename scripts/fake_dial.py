"""Stage-4 ops verification: inject one DialRequest into ``engine:dial``.

Usage::

    python -m scripts.fake_dial \
      --db-url postgresql+asyncpg://bears@localhost:5432/isales_dev \
      --redis-url redis://localhost:6379/0 \
      --campaign-id 1

If ``--lead-id`` is omitted the script creates a fresh lead with
``--phone-number`` (default ``+8613800000000``). It then INCRs the global
concurrency counter (mirroring scheduler dispatch behaviour) and LPUSHes a
schema-valid ``DialRequest``.

The engine consumes the message via ``dial_loop`` and runs the standard
``run_session`` flow with mock providers + ``MockTelephonyClient``, which
will idle in LISTENING until a real ASR drives speech (or silence triggers
hangup). For end-to-end verification you also feed user audio through the
mock telephony — see ``tests/test_run_loop.py`` for the pattern.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from isales_common.models import Lead
from isales_common.schemas.messages.dial import (
    DialLeadInfo,
    DialRequest,
    PromptVersionsSnapshot,
)
from isales_common.utils.redis import get_redis as common_redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

CONCURRENCY_KEY_DEFAULT = "isales:concurrency:active"
DIAL_QUEUE_DEFAULT = "engine:dial"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject a DialRequest into engine:dial")
    p.add_argument(
        "--db-url",
        required=True,
        help="postgresql+asyncpg://… (used to create / look up the lead)",
    )
    p.add_argument("--redis-url", required=True, help="redis://… for LPUSH + INCR")
    p.add_argument("--campaign-id", type=int, required=True)
    p.add_argument("--lead-id", type=int, default=None)
    p.add_argument("--phone-number", default="+8613800000000")
    p.add_argument("--caller-id", default="+8613900000000")
    p.add_argument("--device-id", type=int, default=1)
    p.add_argument("--queue", default=DIAL_QUEUE_DEFAULT)
    p.add_argument("--concurrency-key", default=CONCURRENCY_KEY_DEFAULT)
    p.add_argument(
        "--no-incr",
        action="store_true",
        help="Skip INCR (engine clamps DECR<0 to zero, so this is safe).",
    )
    return p.parse_args()


async def _ensure_lead(
    db_url: str, *, campaign_id: int, lead_id: int | None, phone: str
) -> tuple[int, str]:
    engine = create_async_engine(db_url, future=True)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as db:
            if lead_id is not None:
                lead = await db.get(Lead, lead_id)
                if lead is None:
                    raise SystemExit(f"lead {lead_id} not found")
                return lead.id, lead.phone

            lead = Lead(campaign_id=campaign_id, phone=phone)
            db.add(lead)
            await db.commit()
            await db.refresh(lead)
            return lead.id, lead.phone
    finally:
        await engine.dispose()


async def _inject(args: argparse.Namespace) -> None:
    lead_id, phone = await _ensure_lead(
        args.db_url,
        campaign_id=args.campaign_id,
        lead_id=args.lead_id,
        phone=args.phone_number,
    )

    request = DialRequest(
        lead=DialLeadInfo(
            lead_id=lead_id,
            campaign_id=args.campaign_id,
            phone=phone,
        ),
        history=[],
        prompt_versions=PromptVersionsSnapshot(),
        caller_id=args.caller_id,
        device_id=args.device_id,
    )

    redis: Any = common_redis(args.redis_url)
    try:
        if not args.no_incr:
            await redis.incr(args.concurrency_key)
        await redis.lpush(args.queue, request.model_dump_json())
        print(
            f"injected DialRequest lead_id={lead_id} campaign_id={args.campaign_id} "
            f"queue={args.queue} message_id={request.message_id}"
        )
    finally:
        await redis.close()


def main() -> None:
    args = _parse_args()
    asyncio.run(_inject(args))


if __name__ == "__main__":
    main()
