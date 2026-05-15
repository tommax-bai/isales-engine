"""JwtTokenVerifier — verify tokens against the shared edge-token claim shape.

Claim shape mirrors :mod:`isales_api.edge_token`; mint logic is inlined to
keep this test independent of the isales-api repo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from isales_common.transport.cloud_edge import InvalidToken
from jose import jwt as jose_jwt

from isales_engine.transport.jwt_token_verifier import (
    ALGORITHM,
    AUDIENCE,
    JwtTokenVerifier,
)

SECRET = "test-secret-32-bytes-or-more-please-yes"


def _mint(
    sub: str = "edge-1",
    *,
    secret: str = SECRET,
    ttl: timedelta = timedelta(hours=1),
    **overrides: Any,
) -> str:
    now = datetime.now(tz=UTC)
    claims: dict[str, Any] = {
        "sub": sub,
        "aud": AUDIENCE,
        "scope": "edge",
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    claims.update(overrides)
    token: str = jose_jwt.encode(claims, secret, algorithm=ALGORITHM)
    return token


@pytest.mark.asyncio
async def test_verifier_accepts_well_formed_token() -> None:
    verifier = JwtTokenVerifier(secret=SECRET)

    identity = await verifier.verify(_mint("edge-1"))

    assert identity.edge_device_id == "edge-1"
    assert identity.tenant_id is None


@pytest.mark.asyncio
async def test_verifier_rejects_wrong_secret() -> None:
    verifier = JwtTokenVerifier(secret="different-secret-32-bytes-or-more-yes")

    with pytest.raises(InvalidToken):
        await verifier.verify(_mint())


@pytest.mark.asyncio
async def test_verifier_rejects_wrong_audience() -> None:
    verifier = JwtTokenVerifier(secret=SECRET)

    with pytest.raises(InvalidToken):
        await verifier.verify(_mint(aud="frontend"))


@pytest.mark.asyncio
async def test_verifier_rejects_wrong_scope() -> None:
    verifier = JwtTokenVerifier(secret=SECRET)

    with pytest.raises(InvalidToken):
        await verifier.verify(_mint(scope="user"))


@pytest.mark.asyncio
async def test_verifier_rejects_expired_token() -> None:
    verifier = JwtTokenVerifier(secret=SECRET)

    with pytest.raises(InvalidToken):
        await verifier.verify(_mint(ttl=timedelta(seconds=-1)))


@pytest.mark.asyncio
async def test_verifier_rejects_garbage() -> None:
    verifier = JwtTokenVerifier(secret=SECRET)

    with pytest.raises(InvalidToken):
        await verifier.verify("not-a-jwt")


@pytest.mark.asyncio
async def test_verifier_rejects_empty_sub() -> None:
    verifier = JwtTokenVerifier(secret=SECRET)

    with pytest.raises(InvalidToken):
        await verifier.verify(_mint(sub=""))


def test_verifier_requires_non_empty_secret() -> None:
    with pytest.raises(ValueError):
        JwtTokenVerifier(secret="")
