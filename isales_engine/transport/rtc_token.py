"""Aliyun RTC token signing (cloud-side only).

NOTE on naming: "ARTC" in Aliyun docs is the SDK family name
(ApsaraVideo Real-time Communication SDK) used under two product lines:
(1) Aliyun RTC PaaS — two-way audio/video with channels, publish/subscribe;
(2) ApsaraVideo Live — one-to-many low-latency live streaming.
iSales targets (1) — Aliyun RTC PaaS. The PaaS-vs-Live distinction shows
up in token format: v3.0 binary (this module) vs the deprecated
``sha256(appid+appkey+ch+uid+nonce+ts)`` Live algorithm, which the RTC
roomserver rejects with HTTP 401.

Spec: arch-cloud-edge-split / design.md Decision 1 (Aliyun RTC PaaS) +
Decision 5 (control plane); device-hardware § Requirement: 云端 engine 的
ARTC SDK 接入 — "云端 engine 自己生成 token + 通过 cloud-edge gRPC
DialCommand.rtc_token 下发给边缘".

Algorithm: authoritative Python 3 reference is
``github.com/aliyun/AliRTCSample/Server/python3/`` (linked from
help.aliyun.com/zh/document_detail/2689025.html — the only correct doc).
The simpler ``sha256(appid+appkey+ch+uid+nonce+ts)`` algorithm published
elsewhere (e.g. the old ``aliyunvideo/AliRtcAppServer`` repo) is for the
deprecated 直播 (Live) product and is rejected by Aliyun RTC roomserver
with HTTP 401 "auth invalid" (实测踩坑过 2026-05-24)。

Token format::

    VERSION_0 ("000") + base64(zlib(_pad256(uint32(sig_len) || sig || _pad256(body))))

Body layout (big-endian)::

    uint32 app_id_len + app_id_bytes
    uint32 issue_timestamp
    uint32 salt
    uint32 expire_timestamp
    service:
        uint32 channel_id_len + channel_id_bytes
        uint32 user_id_len + user_id_bytes
        bool has_privilege [+ uint32 privilege]
    options:
        bool has_engine_options [+ uint32 count + (uint32 klen + k + uint32 vlen + v) ...]

Signing key derivation (sign_utils.generate_sign)::

    sign_key_1 = HMAC-SHA256(key=uint32(issue_ts), msg=app_key)
    sign_key   = HMAC-SHA256(key=uint32(salt),    msg=sign_key_1)

Signature::

    sig = HMAC-SHA256(key=sign_key, msg=pad256(body))

The ``_pad256`` helper pads to the smallest power-of-2 multiple of 256
≥ len(input) (e.g. 200B → 256B; 300B → 512B; 600B → 1024B).

No ``nonce`` field — entropy is the random ``salt`` (uint32, embedded
in the body and hashed into the signing key derivation).

Timestamps are Unix epoch **seconds**. Token TTL is capped at 24 h
server-side per Aliyun spec; this module defaults to 12 h.

This module is the ONLY place in the iSales codebase that accepts the
RTC AppKey. AppKey is a cloud-only secret (loaded from the
``ISALES_RTC_APP_KEY`` env var on the engine ECS); it MUST NOT cross the
cloud-edge boundary. Edges receive only the **signed token** via
:class:`cloud_edge_pb2.DialCommand.rtc_token`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import secrets
import struct
import time
import zlib
from dataclasses import dataclass, field


_VERSION_0 = "000"

# Privilege bits (Service.add_*_publish_privilege). v1.0 Aliyun RTC iSales
# 用 audio-only push/pull，留 None (= ENABLE_PRIVILEGE 默认 = allow all)
# 跟 sample example0 ("full privileges as default") 行为一致。
_ENABLE_PRIVILEGE = 1
_ENABLE_AUDIO_PRIVILEGE = 2
_ENABLE_VIDEO_PRIVILEGE = 4
_ENABLE_SCREEN_PRIVILEGE = 8


@dataclass(frozen=True)
class RtcCredentials:
    """Signed RTC token + the metadata an SDK ``JoinChannel`` call needs.

    No ``nonce`` field — v3.0 binary token format doesn't use one; entropy
    is the internal random ``salt`` (not surfaced to callers).
    """

    app_id: str
    channel: str
    user_id: str
    token: str
    expires_at: int  # Unix epoch seconds; the moment the token stops working

    # AuthInfo struct still needs a nonce field — SDK header AliEngineAuthInfo
    # has ``char* nonce`` even though the binary token format embeds entropy
    # internally. Pass empty string; SDK won't recompute hash from it.
    nonce: str = ""

    # Optional: pin issue_ts/salt for reproducible-token tests.
    issue_ts: int = field(default=0, compare=False)
    salt: int = field(default=0, compare=False)


def _pack_u32(n: int) -> bytes:
    return struct.pack(">I", n)


def _next_pow2_multiple_of_256(n: int) -> int:
    if n <= 0:
        return 0
    fixed = 256
    while fixed < n:
        fixed *= 2
    return fixed


def _pad256(src: bytes) -> bytes:
    target = _next_pow2_multiple_of_256(len(src))
    if len(src) >= target:
        return src[:target]
    return src + b"\x00" * (target - len(src))


def _zlib_compress(data: bytes) -> bytes:
    c = zlib.compressobj(wbits=zlib.MAX_WBITS)
    return c.compress(data) + c.flush()


def _generate_sign_key(app_key: str, issue_ts: int, salt: int) -> bytes:
    """Two-step HMAC-SHA256 key derivation per sign_utils.generate_sign."""
    sign_key_1 = hmac.new(
        _pack_u32(issue_ts), app_key.encode("utf-8"), hashlib.sha256
    ).digest()
    return hmac.new(_pack_u32(salt), sign_key_1, hashlib.sha256).digest()


def _pack_service(channel_id: str, user_id: str, privilege: int | None) -> bytes:
    buf = io.BytesIO()
    cid = channel_id.encode("utf-8")
    buf.write(_pack_u32(len(cid)))
    buf.write(cid)
    uid = user_id.encode("utf-8")
    buf.write(_pack_u32(len(uid)))
    buf.write(uid)
    if privilege is not None:
        buf.write(struct.pack(">?I", True, privilege))
    else:
        buf.write(struct.pack(">?", False))
    return buf.getvalue()


def _pack_options(engine_options: dict[str, str] | None) -> bytes:
    # Vendor's reference impl distinguishes ``None`` (omit options block, 1
    # byte ``False``) from an empty dict (emit block header with count 0, 5
    # bytes ``True`` + ``uint32(0)``). DingRTC 3.x's GSLB verifier — and its
    # console "Token生成器" — defaults engine_options to an empty dict, so a
    # ``None`` here produces a token whose body bytes differ from vendor's by
    # 4 bytes, breaking the HMAC signature. (Caught 2026-05-26 via byte-diff
    # of console-issued token vs. self-signed token for same channel/user.)
    buf = io.BytesIO()
    if engine_options is None:
        buf.write(struct.pack(">?", False))
        return buf.getvalue()
    buf.write(struct.pack(">?I", True, len(engine_options)))
    for k, v in sorted(engine_options.items()):
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        buf.write(_pack_u32(len(kb)))
        buf.write(kb)
        buf.write(_pack_u32(len(vb)))
        buf.write(vb)
    return buf.getvalue()


class RtcTokenIssuer:
    """Signs Aliyun RTC tokens using the AppKey.

    Designed to be constructed once per engine process from env vars and
    re-used for every call. Stateless aside from the bound AppId / AppKey.

    Usage::

        issuer = RtcTokenIssuer(
            app_id=os.environ["ISALES_RTC_APP_ID"],
            app_key=os.environ["ISALES_RTC_APP_KEY"],
        )
        engine_creds, edge_creds = issuer.sign_for_call(call_id="c-001")

    Token TTL defaults to 12 h (sample example uses 12 h, max is 24 h).
    """

    def __init__(
        self,
        *,
        app_id: str,
        app_key: str,
        default_ttl_seconds: int = 12 * 3600,
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
        return self._app_id

    def sign(
        self,
        *,
        channel: str,
        user_id: str,
        ttl_seconds: int | None = None,
        now: int | None = None,
        issue_ts: int | None = None,
        salt: int | None = None,
        # Default ``{}`` (not ``None``) so the token body emits the options
        # block header — vendor's GSLB requires this. See _pack_options docstring.
        engine_options: dict[str, str] | None = {},  # noqa: B006 -- frozen by `sign`
        privilege: int | None = None,
        # legacy compat: callers from the old algorithm may pass nonce —
        # ignored (no longer in the protocol). Keep param so we don't
        # break callers mid-deploy.
        nonce: str | None = None,
    ) -> RtcCredentials:
        if not channel:
            raise ValueError("channel must be non-empty")
        if not user_id:
            raise ValueError("user_id must be non-empty")
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        if effective_ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        del nonce  # accepted for backwards compat, not used

        now_ts = int(now) if now is not None else int(time.time())
        expires_at = now_ts + effective_ttl
        issue_ts_v = int(issue_ts) if issue_ts is not None else now_ts
        # salt MUST be in [1, issue_ts] per sample (random.randint(1,
        # issue_timestamp)). 用 secrets.randbelow + 1 取 [1, issue_ts]。
        salt_v = (
            int(salt) if salt is not None
            else (secrets.randbelow(max(issue_ts_v, 1)) + 1)
        )

        token = self._build_token(
            channel=channel,
            user_id=user_id,
            expires_at=expires_at,
            issue_ts=issue_ts_v,
            salt=salt_v,
            engine_options=engine_options,
            privilege=privilege,
        )
        return RtcCredentials(
            app_id=self._app_id,
            channel=channel,
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            issue_ts=issue_ts_v,
            salt=salt_v,
        )

    def sign_for_call(
        self,
        call_id: str,
        *,
        ttl_seconds: int | None = None,
    ) -> tuple[RtcCredentials, RtcCredentials]:
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

    def _build_token(
        self,
        *,
        channel: str,
        user_id: str,
        expires_at: int,
        issue_ts: int,
        salt: int,
        engine_options: dict[str, str] | None,
        privilege: int | None,
    ) -> str:
        sign_key = _generate_sign_key(self._app_key, issue_ts, salt)

        body = io.BytesIO()
        aid = self._app_id.encode("utf-8")
        body.write(_pack_u32(len(aid)))
        body.write(aid)
        body.write(_pack_u32(issue_ts))
        body.write(_pack_u32(salt))
        body.write(_pack_u32(expires_at))
        body.write(_pack_service(channel, user_id, privilege))
        body.write(_pack_options(engine_options))

        body_padded = _pad256(body.getvalue())
        signature = hmac.new(sign_key, body_padded, hashlib.sha256).digest()

        outer = io.BytesIO()
        outer.write(_pack_u32(len(signature)))
        outer.write(signature)
        outer.write(body_padded)

        outer_padded = _pad256(outer.getvalue())
        return _VERSION_0 + base64.b64encode(_zlib_compress(outer_padded)).decode("ascii")


__all__ = ["RtcCredentials", "RtcTokenIssuer"]
