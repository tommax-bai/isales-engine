"""Async Redis client wrapper."""

from __future__ import annotations

from typing import Any

from isales_common.utils.redis import get_redis as _get_redis
from redis.asyncio import Redis


def get_redis(url: str, *, decode_responses: bool = True) -> Redis[Any]:
    return _get_redis(url, decode_responses=decode_responses)
