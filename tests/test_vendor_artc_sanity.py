"""Sanity tests for the Aliyun ARTC vendor SDK adaptor.

These tests do NOT exercise real RTC joining — that requires the Linux
``AliRtcCoreService`` binary and a valid AppId. They verify:

1. :func:`load_vendor_modules` finds the SDK when ``ISALES_RTC_SDK_PATH``
   points at the unpacked ``Python/`` directory.
2. :func:`vendor_channel_factory` constructs a working
   :class:`_AliyunArtcChannel` and the resulting object exposes every
   method the :class:`SdkChannel` protocol requires.
3. :class:`SdkLoadError` is raised with an actionable message when the
   path is unset / wrong.

The Python wrapper modules (``AliRTCEngine``, ``AliRTCLinuxSdkDefine``)
are pure-Python and import on any platform; only an actual
``CreateAliRTCEngine`` call drags in the Linux-only ``.so``. So these
tests run on macOS / CI / Linux equally.

To run locally::

    ISALES_RTC_SDK_PATH=/Users/.../AliRTCSDK_Linux-7.10.2/Python \\
        pytest tests/test_vendor_artc_sanity.py

When the env is unset the load test is skipped (so CI doesn't fail on
machines that don't have the tarball unpacked).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from isales_engine.transport._rtc_sdk import (
    SdkLoadError,
    _AliyunArtcChannel,
    load_vendor_modules,
    vendor_channel_factory,
)


def _sdk_path_or_skip() -> str:
    path = os.environ.get("ISALES_RTC_SDK_PATH")
    if not path:
        pytest.skip("ISALES_RTC_SDK_PATH not set; vendor SDK not unpacked")
    if not Path(path, "AliRTCEngine.py").exists():
        pytest.skip(f"AliRTCEngine.py not found under {path}")
    return path


def test_load_vendor_modules_imports_engine_and_define() -> None:
    _sdk_path_or_skip()
    engine_mod, define_mod = load_vendor_modules()

    # Engine module: factory + handler base class are what we depend on.
    assert hasattr(engine_mod, "CreateAliRTCEngine")
    assert hasattr(engine_mod, "EngineEventHandlerInterface")

    # Define module: the value classes we consume.
    assert hasattr(define_mod, "AuthInfo")
    assert hasattr(define_mod, "JoinChannelConfig")
    assert hasattr(define_mod, "AudioFormat")
    assert hasattr(define_mod, "AliEngineClientRole")


def test_vendor_channel_factory_builds_aliyun_artc_channel() -> None:
    _sdk_path_or_skip()
    channel = vendor_channel_factory(app_id="sanity-app")
    assert isinstance(channel, _AliyunArtcChannel)

    # Structural SdkChannel surface — duck-check rather than runtime
    # isinstance (Protocol with non-runtime_checkable subclassing leaks).
    for method in (
        "join",
        "leave",
        "push_audio",
        "on_inbound_frame",
        "on_buffer_state",
    ):
        assert callable(getattr(channel, method)), method


def test_handler_subclass_routes_events_to_channel() -> None:
    """The vendor's EngineEventHandlerInterface subclass we build must
    forward the four callbacks we care about to the channel's stored
    callbacks.

    We trigger callbacks by constructing the handler directly and
    invoking its methods (no native binary involved). This is the only
    place the adaptor's event-routing logic gets unit-tested; the
    Linux-only ``CreateAliRTCEngine`` path is covered by integration
    tests on the cloud ECS.
    """
    _sdk_path_or_skip()
    channel = vendor_channel_factory(app_id="sanity-app")
    assert isinstance(channel, _AliyunArtcChannel)

    inbound: list[tuple[str, bytes, int, int]] = []
    buffer_states: list[bool] = []

    def inbound_cb(uid: str, pcm: bytes, sr: int, ts: int) -> None:
        inbound.append((uid, pcm, sr, ts))

    def buffer_cb(is_full: bool) -> None:
        buffer_states.append(is_full)

    channel.on_inbound_frame(inbound_cb)
    channel.on_buffer_state(buffer_cb)

    # Build the handler subclass directly (bypass real engine).
    handler = channel._build_handler()  # noqa: SLF001 — testing internal wiring

    # Buffer-full / drained → buffer_cb invoked with the bool.
    handler.OnPushAudioFrameBufferFull(True)
    handler.OnPushAudioFrameBufferFull(False)
    assert buffer_states == [True, False]

    # OnJoinChannelResult → sets the channel's join event.
    assert not channel._join_event.is_set()  # noqa: SLF001
    handler.OnJoinChannelResult(0, "c-1", "engine-c-1")
    assert channel._join_event.is_set()  # noqa: SLF001
    assert channel._join_result == 0  # noqa: SLF001

    # OnSubscribeAudioFrame with a fake AudioFrame-like object.
    class _FakePcm:
        pcmBuf_ = b"\x01\x02\x03\x04"
        sample_rates_ = 16000

    class _FakeFrame:
        pcm = _FakePcm()

    handler.OnSubscribeAudioFrame("edge-c-1", _FakeFrame())
    assert len(inbound) == 1
    uid, pcm, sr, ts = inbound[0]
    assert uid == "edge-c-1"
    assert pcm == b"\x01\x02\x03\x04"
    assert sr == 16000
    assert ts == 0


def test_sdk_load_error_when_path_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointing ISALES_RTC_SDK_PATH at a non-existent dir raises
    SdkLoadError with an actionable message.

    Requires scrubbing both ``sys.modules`` (so importlib re-attempts the
    import) and ``sys.path`` (so any previously-added real SDK directory
    doesn't satisfy the import from a different env). The fixture restores
    everything after the assertion.
    """
    import sys

    orig_path = sys.path[:]
    orig_modules = {
        name: sys.modules[name]
        for name in ("AliRTCEngine", "AliRTCLinuxSdkDefine")
        if name in sys.modules
    }
    try:
        # Remove any path entry that still contains the vendor modules,
        # so the bogus env path is the only candidate importlib sees.
        for name in orig_modules:
            del sys.modules[name]
        sys.path[:] = [
            p for p in sys.path if not Path(p, "AliRTCEngine.py").exists()
        ]

        monkeypatch.setenv("ISALES_RTC_SDK_PATH", "/nonexistent/path/here")
        with pytest.raises(SdkLoadError) as exc_info:
            load_vendor_modules()
        msg = str(exc_info.value)
        assert "ISALES_RTC_SDK_PATH" in msg
        assert "install-artc-sdk.sh" in msg
    finally:
        sys.path[:] = orig_path
        sys.modules.update(orig_modules)
