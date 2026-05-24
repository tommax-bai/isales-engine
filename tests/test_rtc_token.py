"""RTC token issuer tests.

Spec: arch-cloud-edge-split / device-hardware § Requirement: 云端 engine 的
ARTC SDK 接入.

Algorithm is the v3.0 binary format from authoritative sample
``github.com/aliyun/AliRTCSample/Server/python3/``. The earlier
sha256(concat) algorithm published in other Aliyun repos is for the
deprecated 直播 (Live) product and is rejected by RTC roomserver
(2026-05-24 实测踩坑)。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import struct
import zlib

import pytest

from isales_engine.transport.rtc_token import (
    RtcCredentials,
    RtcTokenIssuer,
)


# Construction validation


def test_empty_app_id_raises() -> None:
    with pytest.raises(ValueError, match="app_id"):
        RtcTokenIssuer(app_id="", app_key="k")


def test_empty_app_key_raises() -> None:
    with pytest.raises(ValueError, match="app_key"):
        RtcTokenIssuer(app_id="a", app_key="")


def test_non_positive_default_ttl_raises() -> None:
    with pytest.raises(ValueError, match="default_ttl_seconds"):
        RtcTokenIssuer(app_id="a", app_key="k", default_ttl_seconds=0)


# Sign validation


def test_empty_channel_raises() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    with pytest.raises(ValueError, match="channel"):
        issuer.sign(channel="", user_id="u")


def test_empty_user_id_raises() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    with pytest.raises(ValueError, match="user_id"):
        issuer.sign(channel="c", user_id="")


def test_zero_ttl_override_raises() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k", default_ttl_seconds=600)
    with pytest.raises(ValueError, match="ttl_seconds"):
        issuer.sign(channel="c", user_id="u", ttl_seconds=0)


def test_default_ttl_used_when_unspecified() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k", default_ttl_seconds=600)
    creds = issuer.sign(channel="c", user_id="u", now=1000)
    assert creds.expires_at == 1600


# Algorithm correctness — pinned against authoritative sample.


def _reference_token(
    app_id: str, app_key: str, channel: str, user_id: str,
    expire_ts: int, issue_ts: int, salt: int,
) -> str:
    """Re-implementation of aliyun/AliRTCSample/Server/python3 — kept
    inline so the test catches any drift in our :class:`RtcTokenIssuer`
    against the authoritative algorithm."""
    sk1 = hmac.new(struct.pack(">I", issue_ts), app_key.encode(), hashlib.sha256).digest()
    sk = hmac.new(struct.pack(">I", salt), sk1, hashlib.sha256).digest()

    body = io.BytesIO()
    aid = app_id.encode()
    body.write(struct.pack(">I", len(aid))); body.write(aid)
    body.write(struct.pack(">I", issue_ts))
    body.write(struct.pack(">I", salt))
    body.write(struct.pack(">I", expire_ts))
    cb = channel.encode()
    body.write(struct.pack(">I", len(cb))); body.write(cb)
    ub = user_id.encode()
    body.write(struct.pack(">I", len(ub))); body.write(ub)
    body.write(struct.pack(">?", False))  # no privilege
    body.write(struct.pack(">?", False))  # no engine_options

    def pad256(b: bytes) -> bytes:
        n = len(b)
        if n == 0:
            return b
        s = 256
        while s < n:
            s *= 2
        return b + b"\x00" * (s - n)

    bp = pad256(body.getvalue())
    sig = hmac.new(sk, bp, hashlib.sha256).digest()
    outer = io.BytesIO()
    outer.write(struct.pack(">I", len(sig))); outer.write(sig); outer.write(bp)
    op = pad256(outer.getvalue())

    c = zlib.compressobj(wbits=zlib.MAX_WBITS)
    compressed = c.compress(op) + c.flush()
    return "000" + base64.b64encode(compressed).decode()


def test_token_matches_official_sample_byte_for_byte() -> None:
    issuer = RtcTokenIssuer(app_id="my-app", app_key="secret-key")
    creds = issuer.sign(
        channel="call-001",
        user_id="engine-call-001",
        now=1_700_000_000,
        ttl_seconds=3600,
        issue_ts=1_700_000_000,
        salt=42,
    )
    expected = _reference_token(
        "my-app", "secret-key", "call-001", "engine-call-001",
        expire_ts=1_700_003_600, issue_ts=1_700_000_000, salt=42,
    )
    assert creds.token == expected
    assert creds.token.startswith("000")
    assert creds.expires_at == 1_700_003_600
    assert creds.app_id == "my-app"
    assert creds.channel == "call-001"
    assert creds.user_id == "engine-call-001"
    assert creds.issue_ts == 1_700_000_000
    assert creds.salt == 42


def test_pinned_inputs_produce_pinned_token() -> None:
    """同 channel+uid+now+issue_ts+salt → 同 token (no hidden state)."""
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    a = issuer.sign(channel="c", user_id="u", now=100, ttl_seconds=10, issue_ts=99, salt=7)
    b = issuer.sign(channel="c", user_id="u", now=100, ttl_seconds=10, issue_ts=99, salt=7)
    assert a.token == b.token


def test_random_salt_produces_distinct_tokens() -> None:
    """Two sign() calls with no salt override get different tokens —
    salt is random per call, sourced from secrets."""
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    a = issuer.sign(channel="c", user_id="u", now=100)
    b = issuer.sign(channel="c", user_id="u", now=100)
    # salts almost certainly differ (1 in issue_ts ≈ 1 in 100 chance of
    # collision; flaky risk negligible).
    assert a.salt != b.salt or a.token != b.token


def test_app_key_never_leaked_via_creds() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="VERY-SECRET-KEY")
    creds = issuer.sign(channel="c", user_id="u")
    assert not hasattr(creds, "app_key")
    rendered = repr(creds)
    assert "VERY-SECRET-KEY" not in rendered


# sign_for_call


def test_sign_for_call_returns_engine_and_edge_pair() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    engine_creds, edge_creds = issuer.sign_for_call("call-007")
    assert engine_creds.channel == "call-007"
    assert engine_creds.user_id == "engine-call-007"
    assert edge_creds.channel == "call-007"
    assert edge_creds.user_id == "edge-call-007"
    assert engine_creds.token != edge_creds.token


def test_sign_for_call_empty_call_id_raises() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    with pytest.raises(ValueError, match="call_id"):
        issuer.sign_for_call("")


# Roundtrip-shape sanity


def test_token_is_base64_after_version_prefix() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    creds = issuer.sign(channel="c", user_id="u", now=1_700_000_000, ttl_seconds=3600)
    assert creds.token.startswith("000")
    # base64 portion decodes cleanly + gzlib-decompresses.
    body = base64.b64decode(creds.token[3:])
    decompressed = zlib.decompress(body, zlib.MAX_WBITS)
    # Signature is 32 bytes (HMAC-SHA256 output) → header is 4-byte big-endian 32.
    assert decompressed[:4] == struct.pack(">I", 32)


# RtcCredentials immutability


def test_creds_is_immutable() -> None:
    creds = RtcCredentials(
        app_id="a",
        channel="c",
        user_id="u",
        token="000xxx",
        expires_at=0,
    )
    with pytest.raises(AttributeError):
        creds.token = "tampered"  # type: ignore[misc]
