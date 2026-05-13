"""RTC token issuer tests.

Spec: arch-cloud-edge-split / device-hardware § Requirement: 云端 engine 的
ARTC SDK 接入.

Algorithm cross-check uses the published formula from
https://help.aliyun.com/document_detail/159037.html. We verify the SHA-256
ordering with a hand-computed expected value rather than re-deriving the
formula in the test (otherwise the test just mirrors the implementation
and catches nothing).
"""

from __future__ import annotations

import hashlib

import pytest

from isales_engine.transport.rtc_token import (
    RtcCredentials,
    RtcTokenIssuer,
    _generate_nonce,
)

# --------------------------------------------------------------------------
# Construction validation
# --------------------------------------------------------------------------


def test_empty_app_id_raises() -> None:
    with pytest.raises(ValueError, match="app_id"):
        RtcTokenIssuer(app_id="", app_key="k")


def test_empty_app_key_raises() -> None:
    with pytest.raises(ValueError, match="app_key"):
        RtcTokenIssuer(app_id="a", app_key="")


def test_non_positive_default_ttl_raises() -> None:
    with pytest.raises(ValueError, match="default_ttl_seconds"):
        RtcTokenIssuer(app_id="a", app_key="k", default_ttl_seconds=0)


# --------------------------------------------------------------------------
# Algorithm correctness
# --------------------------------------------------------------------------


def test_token_matches_aliyun_sha256_formula() -> None:
    """Independently compute the expected token and compare.

    The doc says::

        token = sha256(app_id + app_key + channel_id + user_id + nonce + timestamp).hexdigest()

    Pin every input so the assertion is reproducible.
    """
    issuer = RtcTokenIssuer(app_id="my-app", app_key="secret-key")
    creds = issuer.sign(
        channel="call-001",
        user_id="engine-call-001",
        now=1_700_000_000,
        ttl_seconds=3600,
        nonce="AK-abc123",
    )

    # Hand-rolled expected value following the exact published ordering.
    expected_raw = (
        "my-app" + "secret-key" + "call-001" + "engine-call-001" + "AK-abc123" + "1700003600"
    )
    expected = hashlib.sha256(expected_raw.encode("utf-8")).hexdigest()

    assert creds.token == expected
    assert creds.expires_at == 1_700_003_600
    assert creds.app_id == "my-app"
    assert creds.channel == "call-001"
    assert creds.user_id == "engine-call-001"
    assert creds.nonce == "AK-abc123"


def test_same_inputs_same_token() -> None:
    """Identical inputs (including nonce + now) MUST produce identical
    tokens. Catches accidental state leaks between sign calls."""
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    a = issuer.sign(channel="c", user_id="u", now=100, ttl_seconds=10, nonce="AK-x")
    b = issuer.sign(channel="c", user_id="u", now=100, ttl_seconds=10, nonce="AK-x")
    assert a.token == b.token


def test_different_nonce_yields_different_token() -> None:
    """Two signings of the same call/uid get different tokens because
    the random nonce differs — this is what makes replay attacks hard."""
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    a = issuer.sign(channel="c", user_id="u", now=100)
    b = issuer.sign(channel="c", user_id="u", now=100)
    assert a.nonce != b.nonce
    assert a.token != b.token


def test_app_key_never_leaked_via_creds() -> None:
    """RtcCredentials MUST NOT expose the AppKey — it's a cloud-only
    secret that flows only into the SHA256 input, never out."""
    issuer = RtcTokenIssuer(app_id="a", app_key="VERY-SECRET-KEY")
    creds = issuer.sign(channel="c", user_id="u")

    # The dataclass exposes app_id but not app_key.
    assert not hasattr(creds, "app_key")
    # And the AppKey must not be reconstructable from the token surface
    # (sha256 is a one-way function, but assert the obvious: the literal
    # string isn't anywhere in the serialised creds).
    rendered = repr(creds)
    assert "VERY-SECRET-KEY" not in rendered


# --------------------------------------------------------------------------
# Sign validation
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Convenience: sign_for_call
# --------------------------------------------------------------------------


def test_sign_for_call_returns_engine_and_edge_pair() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    engine_creds, edge_creds = issuer.sign_for_call("call-007")
    assert engine_creds.channel == "call-007"
    assert engine_creds.user_id == "engine-call-007"
    assert edge_creds.channel == "call-007"
    assert edge_creds.user_id == "edge-call-007"


def test_sign_for_call_engine_and_edge_have_independent_nonces() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    engine_creds, edge_creds = issuer.sign_for_call("c-1")
    # Each is signed independently with its own freshly-generated nonce.
    assert engine_creds.nonce != edge_creds.nonce
    assert engine_creds.token != edge_creds.token


def test_sign_for_call_empty_call_id_raises() -> None:
    issuer = RtcTokenIssuer(app_id="a", app_key="k")
    with pytest.raises(ValueError, match="call_id"):
        issuer.sign_for_call("")


# --------------------------------------------------------------------------
# Nonce contract
# --------------------------------------------------------------------------


def test_generated_nonce_has_required_prefix() -> None:
    nonce = _generate_nonce()
    assert nonce.startswith("AK-")


def test_generated_nonce_body_is_alphanumeric() -> None:
    nonce = _generate_nonce()
    body = nonce.removeprefix("AK-")
    assert body.isalnum()
    assert len(body) == 16  # default


def test_generated_nonces_are_unique_across_calls() -> None:
    """Random nonces from N calls SHOULD all differ (probabilistically
    almost certainly with the 62**16 alphabet). If this test ever fails,
    investigate whether the RNG seed got pinned."""
    sample = {_generate_nonce() for _ in range(64)}
    assert len(sample) == 64


def test_nonce_within_64_byte_aliyun_limit() -> None:
    """Aliyun spec says nonce ≤ 64 bytes total. With 'AK-' prefix + 16
    char body that's 19 bytes — well within the budget."""
    nonce = _generate_nonce()
    assert len(nonce.encode("utf-8")) <= 64


# --------------------------------------------------------------------------
# RtcCredentials immutability
# --------------------------------------------------------------------------


def test_creds_is_immutable() -> None:
    creds = RtcCredentials(
        app_id="a",
        channel="c",
        user_id="u",
        nonce="AK-x",
        token="t",
        expires_at=0,
    )
    with pytest.raises(AttributeError):
        creds.token = "tampered"  # type: ignore[misc]  # testing immutability
