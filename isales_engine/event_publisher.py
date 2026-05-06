"""Fire-and-forget EngineEvent publisher.

Spec: service-communication § 通信通道矩阵 (engine→api row);
      message-contract § Scenario "v1 必有的消息类".

Pub/Sub semantics: occasional message loss is acceptable so we never block
the call's main path on a publish. Each ``publish`` schedules an internal
task with a 2-second budget and logs warnings on timeout / errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from isales_common.schemas.messages.engine_event import EngineEvent
from pydantic import TypeAdapter
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

CHANNEL_TEMPLATE = "engine:events:campaign:{campaign_id}"

_event_adapter: TypeAdapter[EngineEvent] = TypeAdapter(EngineEvent)


class EventPublisher:
    def __init__(
        self,
        redis: Redis[Any],
        *,
        publish_timeout_s: float = 2.0,
    ) -> None:
        self._redis = redis
        self._publish_timeout_s = publish_timeout_s
        self._inflight: set[asyncio.Task[None]] = set()

    def publish(self, campaign_id: int, event: EngineEvent) -> None:
        """Schedule a non-blocking publish. Returns immediately."""

        task = asyncio.create_task(
            self._publish(campaign_id, event), name="engine_event_publish"
        )
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def drain(self) -> None:
        """Wait for all in-flight publishes (used at graceful shutdown)."""

        if not self._inflight:
            return
        await asyncio.gather(*self._inflight, return_exceptions=True)

    async def _publish(self, campaign_id: int, event: EngineEvent) -> None:
        channel = CHANNEL_TEMPLATE.format(campaign_id=campaign_id)
        payload = _event_adapter.dump_json(event).decode()
        try:
            async with asyncio.timeout(self._publish_timeout_s):
                await self._redis.publish(channel, payload)
        except TimeoutError:
            logger.warning(
                "engine_event_publish_timeout campaign_id=%s type=%s",
                campaign_id,
                getattr(event, "type", "?"),
            )
        except Exception:
            logger.exception(
                "engine_event_publish_failed campaign_id=%s type=%s",
                campaign_id,
                getattr(event, "type", "?"),
            )
