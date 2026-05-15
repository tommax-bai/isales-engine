"""HS256 JWT bearer-token verifier for the cloud-edge gRPC stream.

Spec: arch-cloud-edge-split § service-communication Requirement "云-边控制面"
      Scenario "gRPC 鉴权" — token signed by isales-api with the shared
      ``ISALES_JWT_SECRET`` but a distinct ``aud=cloud-edge`` audience.

Companion to :mod:`isales_api.edge_token` which mints the token. This module
verifies one on the engine side and returns an :class:`EdgeIdentity` bound to
the JWT's ``sub`` claim.

Claim contract (must match :mod:`isales_api.edge_token`)::

    sub  = edge_device_id   (cloud-side PK; engine binds the stream)
    aud  = ``cloud-edge``   (audience isolation from frontend JWTs)
    scope = ``edge``        (defence-in-depth so a frontend JWT is rejected)
    iat / exp               (HS256 standard claims; exp enforced)
"""

from __future__ import annotations

from isales_common.transport.cloud_edge import (
    EdgeIdentity,
    InvalidToken,
    TokenVerifier,
)
from jose import JWTError
from jose import jwt as jose_jwt

ALGORITHM = "HS256"
AUDIENCE = "cloud-edge"
SCOPE = "edge"


class JwtTokenVerifier(TokenVerifier):
    """Verify edge-device JWTs minted by :func:`isales_api.edge_token.mint_edge_token`.

    Construction is intentionally cheap (just stores the secret); the actual
    verification happens on every stream open. C2 will swap this out for a
    rotating tenant-scoped impl with the same surface.
    """

    def __init__(self, *, secret: str, audience: str = AUDIENCE) -> None:
        if not secret:
            raise ValueError("secret must be non-empty")
        self._secret = secret
        self._audience = audience

    async def verify(self, token: str) -> EdgeIdentity:
        try:
            claims = jose_jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],
                audience=self._audience,
            )
        except JWTError as exc:
            raise InvalidToken(str(exc)) from exc

        scope = claims.get("scope")
        if scope != SCOPE:
            raise InvalidToken(f"unexpected scope: {scope!r}")
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub:
            raise InvalidToken("missing or empty sub claim")
        return EdgeIdentity(edge_device_id=sub)


__all__ = ["JwtTokenVerifier"]
