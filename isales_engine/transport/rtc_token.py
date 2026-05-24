"""Aliyun RTC token signing (cloud-side only).

Spec: arch-cloud-edge-split / design.md Decision 1 (Aliyun RTC PaaS) +
Decision 5 (control plane); device-hardware § Requirement: 云端 engine 的
ARTC SDK 接入 — "云端 engine 自己生成 token + 通过 cloud-edge gRPC
DialCommand.rtc_token 下发给边缘".

Algorithm per Aliyun official docs
(https://help.aliyun.com/zh/live/token-based-authentication)::

    token = sha256(app_id + app_key + channel_id + user_id + nonce + timestamp).hexdigest()

The ``timestamp`` is the **expiration** Unix epoch seconds (not issuance
time).

``nonce`` defaults to empty string ``""`` per the current Aliyun doc
(v3 应用 + ARTC SDK 7.x) — "Nonce 推荐为空"。SDK 4-arg
``JoinChannel(token, channel, uid, name)`` form does NOT pass nonce
explicitly, so the SDK server-side validation re-computes hash with
``nonce=""``. If our signer uses any other nonce value (e.g. legacy
"AK-..." prefix used by Aliyun 2.x docs), the hashes diverge and the
service rejects with error 0x01037D81 (discovered 2026-05-24 via
windows-artc-pybind11-join-config §4 实测)。

If a non-empty nonce is required (e.g. SDK 8.x), callers MUST pass it
via the ``JoinChannel(AliEngineAuthInfo, name)`` 2-arg form which
carries the nonce alongside the token; pybind binding does not yet
expose that overload.

This module is the ONLY place in the iSales codebase that accepts the
RTC AppKey. AppKey is a cloud-only secret (loaded from the
``ISALES_RTC_APP_KEY`` env var on the engine ECS); it MUST NOT cross the
cloud-edge boundary. Edges receive only the **signed token** via
:class:`cloud_edge_pb2.DialCommand.rtc_token`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import string
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RtcCredentials:
    """One signed RTC token + the metadata an SDK ``JoinChannel`` call needs.

    Both fields the SDK actually reads (``channel`` / ``user_id`` /
    ``token``) plus bookkeeping (``app_id`` / ``nonce`` / ``expires_at``)
    that simplifies logging, token-refresh logic, and tests.
    """

    app_id: str
    channel: str
    user_id: str
    nonce: str
    token: str
    expires_at: int  # Unix epoch seconds; the moment the token stops working


# Nonce default: empty string (per https://help.aliyun.com/zh/live/
# token-based-authentication "Nonce 推荐为空"). The legacy "AK-xxxx"
# format was from Aliyun 2.x docs; with v3 apps + ARTC SDK 7.x the
# server expects nonce="" for the 4-arg JoinChannel(token, ch, uid,
# name) form. See module docstring.
_NONCE_ALPHABET = string.ascii_letters + string.digits
_NONCE_PREFIX = "AK-"  # kept for explicit-nonce callers (AuthInfo form)


def _generate_nonce(body_length: int = 16) -> str:
    """Legacy "AK-<random>" nonce (Aliyun 2.x docs).

    Kept for callers that explicitly opt in via ``sign(nonce=...)``;
    NOT used as default (default is empty string per current v3 docs).
    """
    body = "".join(secrets.choice(_NONCE_ALPHABET) for _ in range(body_length))
    return _NONCE_PREFIX + body


class RtcTokenIssuer:
    """Signs Aliyun RTC tokens using the AppKey.

    Designed to be constructed once per engine process from env vars and
    re-used for every call. Stateless aside from the bound AppId / AppKey;
    the only "time" input flows through ``sign``'s ``now`` parameter.

    Usage::

        issuer = RtcTokenIssuer(
            app_id=os.environ["ISALES_RTC_APP_ID"],
            app_key=os.environ["ISALES_RTC_APP_KEY"],
        )
        engine_creds, edge_creds = issuer.sign_for_call(call_id="c-001")
        # engine_creds → cloud engine JoinChannel
        # edge_creds.token → Cloud2Edge.DialCommand.rtc_token

    Token TTL: defaults to 24 h, more than enough for any v1.0 call
    duration. The :class:`RtcCredentials.RtcCredentials` (sic) downstream
    proto message carries an ``expires_at`` field for future refresh
    logic, but A2 does not exercise mid-call token refresh.
    """

    def __init__(
        self,
        *,
        app_id: str,
        app_key: str,
        default_ttl_seconds: int = 24 * 3600,
    ) -> None:
        if not app_id:
            raise ValueError("app_id must be non-empty")
        if not app_key:
            raise ValueError("app_key must be non-empty")
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        self._app_id = app_id
        self._app_key = app_key
        self._default_ttl_seconds = default_ttl_seconds

    @property
    def app_id(self) -> str:
        """The bound RTC AppId (safe to expose; AppKey is not)."""
        return self._app_id

    def sign(
        self,
        *,
        channel: str,
        user_id: str,
        ttl_seconds: int | None = None,
        now: int | None = None,
        nonce: str | None = None,
    ) -> RtcCredentials:
        """Sign a token for ``user_id`` to join ``channel``.

        Args:
            channel: RTC channel name. iSales convention: ``call_id``.
            user_id: RTC participant uid, e.g. ``"engine-{call_id}"`` or
                ``"edge-{call_id}"``.
            ttl_seconds: token lifetime in seconds. Defaults to the
                instance's ``default_ttl_seconds``.
            now: override current Unix epoch seconds. Tests use this to
                get deterministic ``expires_at``; production callers leave
                this ``None``.
            nonce: override the nonce. Tests use this for reproducible
                token hashes; production callers leave this ``None`` and
                let the issuer generate a fresh one per call.

        Returns:
            :class:`RtcCredentials` with the signed token + metadata.

        Raises:
            ValueError: if ``channel`` / ``user_id`` is empty, or
                ``ttl_seconds`` is non-positive.
        """
        if not channel:
            raise ValueError("channel must be non-empty")
        if not user_id:
            raise ValueError("user_id must be non-empty")
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        if effective_ttl <= 0:
            raise ValueError("ttl_seconds must be positive")

        now_ts = int(now) if now is not None else int(time.time())
        expires_at = now_ts + effective_ttl
        # Default nonce="" per current Aliyun v3 doc (Base64 Token form).
        nonce_val = nonce if nonce is not None else ""
        sha = self._compute_token(channel, user_id, expires_at)
        # v3 应用 SDK 4-arg JoinChannel 要 Base64(JSON) token (per
        # https://help.aliyun.com/zh/live/token-based-authentication
        # § "Base64 Token")，不是 raw sha256 hex (那是老 v2 form)。
        payload = {
            "appid": self._app_id,
            "channelid": channel,
            "userid": user_id,
            "nonce": nonce_val,
            "timestamp": expires_at,
            "token": sha,
        }
        # NOTE: 不传 separators — Aliyun 官方 Python 示例用 json.dumps()
        # 默认 separator (", ", ": ") 含空格；服务端 byte-match 期望该
        # 格式。紧凑 separators=(",", ":") 会导致 401 auth invalid
        # (实测 2026-05-24 windows-artc-pybind11-join-config §4)。
        token_b64 = base64.b64encode(
            json.dumps(payload).encode("utf-8")
        ).decode("ascii")
        return RtcCredentials(
            app_id=self._app_id,
            channel=channel,
            user_id=user_id,
            nonce=nonce_val,
            token=token_b64,
            expires_at=expires_at,
        )

    def sign_for_call(
        self,
        call_id: str,
        *,
        ttl_seconds: int | None = None,
    ) -> tuple[RtcCredentials, RtcCredentials]:
        """Sign both the engine-side and edge-side credentials for one call.

        Returns ``(engine_creds, edge_creds)``. Channel is the ``call_id``;
        the engine joins as ``"engine-{call_id}"`` and subscribes filtering
        on the edge uid ``"edge-{call_id}"`` — the convention pinned by
        arch-cloud-edge-split design Decision 2.

        The two credentials share the same ``call_id`` channel but have
        independent ``nonce`` / ``token``: each uid signs its own.
        """
        if not call_id:
            raise ValueError("call_id must be non-empty")
        engine_creds = self.sign(
            channel=call_id,
            user_id=f"engine-{call_id}",
            ttl_seconds=ttl_seconds,
        )
        edge_creds = self.sign(
            channel=call_id,
            user_id=f"edge-{call_id}",
            ttl_seconds=ttl_seconds,
        )
        return engine_creds, edge_creds

    def _compute_token(
        self,
        channel: str,
        user_id: str,
        expires_at: int,
    ) -> str:
        # Hash 内**不含 nonce**，per Aliyun v3 doc Python example:
        # h = sha256(); h.update(app_id); h.update(app_key);
        # h.update(channel); h.update(user_id); h.update(timestamp).
        # See https://help.aliyun.com/zh/live/token-based-authentication.
        h = hashlib.sha256()
        h.update(self._app_id.encode("utf-8"))
        h.update(self._app_key.encode("utf-8"))
        h.update(channel.encode("utf-8"))
        h.update(user_id.encode("utf-8"))
        h.update(str(expires_at).encode("utf-8"))
        return h.hexdigest()


__all__ = ["RtcCredentials", "RtcTokenIssuer"]
