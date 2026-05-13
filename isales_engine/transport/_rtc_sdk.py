"""Vendor SDK boundary for the cloud-side RtcSession implementation.

The Aliyun ARTC SDK for Linux Python ships as a binary archive (``.so`` +
Python wrapper) — not on PyPI. To keep the engine code testable and to
absorb minor SDK API drift in one place, we narrow the surface to a small
:class:`SdkChannel` protocol with five methods. Concrete adaptors:

- ``_AliyunArtcChannel`` (production): thin wrapper over the vendor SDK's
  ``Engine`` + ``Channel`` objects, loaded lazily from a vendor-provided
  Python module. Imports the vendor module only when first instantiated, so
  unit tests can run without the vendor binary.
- :class:`isales_engine.transport.aliyun_rtc.InMemorySdkChannel` (tests):
  loopback in-memory implementation. Lives next to the SDK boundary so
  test code doesn't need to redeclare the protocol.

Spec: device-hardware § Requirement: 云端 engine 的 ARTC SDK 接入. The
vendor's actual API names (``JoinChannel``, ``SetExternalAudioSource``,
``PushExternalAudioFrameRawData``, ``OnSubscribeAudioFrame``,
``OnPushAudioFrameBufferFull``) are mapped into the protocol's normalised
verbs at the adaptor layer.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Protocol

# =============================================================================
# Callback signatures
# =============================================================================

#: Called once per inbound audio frame from a remote uid.
#: (uid, pcm, sample_rate, timestamp_ms) → None
InboundFrameCallback = Callable[[str, bytes, int, int], None]

#: Called when the SDK's outbound push buffer transitions full/drained.
#: (is_full,) → None. ``True`` means push is now backpressured.
BufferStateCallback = Callable[[bool], None]


# =============================================================================
# Vendor-agnostic channel protocol
# =============================================================================


class SdkChannel(Protocol):
    """Minimum surface a vendor RTC SDK must expose for cloud-side use.

    Implementations are NOT asyncio-aware; the engine wrapper
    (:class:`AliyunRtcSession`) bridges callbacks to asyncio via
    ``loop.call_soon_threadsafe``.
    """

    def join(
        self,
        channel: str,
        token: str,
        uid: str,
        *,
        send_sample_rate: int,
        send_channels: int,
    ) -> None:
        """Open a channel and join as ``uid`` with ``publisher_subscriber`` role.

        Blocks until the SDK reports the channel joined. Raises on
        auth / network failure.
        """
        ...

    def leave(self) -> None:
        """Leave the channel, release SDK resources. Idempotent."""
        ...

    def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        """Push outbound PCM. Returns 0 on success.

        Returns a positive int when the SDK's outbound buffer is full
        (vendor's ``ERR_AUDIO_BUFFER_FULL``); callers SHOULD await a
        :data:`BufferStateCallback` with ``is_full=False`` before retrying.
        """
        ...

    def on_inbound_frame(self, callback: InboundFrameCallback) -> None:
        """Register the inbound PCM frame callback.

        MUST be called before :meth:`join`. Only one callback is supported.
        """
        ...

    def on_buffer_state(self, callback: BufferStateCallback) -> None:
        """Register the outbound buffer state callback (full/drained).

        MUST be called before :meth:`join`. Only one callback is supported.
        """
        ...


# =============================================================================
# Vendor module loading
# =============================================================================


class SdkLoadError(RuntimeError):
    """Raised when the vendor SDK module cannot be imported.

    Production deploy script (``deploy/cloud/scripts/install-artc-sdk.sh``)
    is responsible for ensuring the vendor module is importable on the
    cloud ECS. Tests pass a pre-built ``SdkChannel`` instance directly to
    :class:`AliyunRtcSession`, bypassing this loader.
    """


# Name of the Python module the vendor's Linux ARTC SDK exposes once
# unpacked from its tarball. Override at deploy time via
# ``ISALES_RTC_SDK_MODULE`` if Aliyun renames the wrapper.
_DEFAULT_SDK_MODULE_ENV = "ISALES_RTC_SDK_MODULE"
_DEFAULT_SDK_MODULE_NAME = "aliyun_artc"


def load_vendor_sdk() -> object:
    """Lazy-import the vendor SDK module (whatever Aliyun ships as the
    Python wrapper around the Linux ARTC SDK binaries).

    Returns the imported module; callers construct a concrete
    :class:`SdkChannel` via :func:`vendor_channel_factory` rather than
    poking at the module directly.

    Raises :class:`SdkLoadError` if the module is not importable.
    """
    module_name = os.environ.get(_DEFAULT_SDK_MODULE_ENV, _DEFAULT_SDK_MODULE_NAME)
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SdkLoadError(
            f"Vendor SDK module {module_name!r} not importable. "
            "Run deploy/cloud/scripts/install-artc-sdk.sh on the cloud ECS, "
            f"or set ${_DEFAULT_SDK_MODULE_ENV} to override the module name."
        ) from exc


def vendor_channel_factory(app_id: str) -> SdkChannel:
    """Construct a production :class:`SdkChannel` for ``app_id``.

    NOT YET IMPLEMENTED — the body adapts whatever the vendor wrapper
    actually exposes (likely ``Engine().CreateChannel()`` or similar) into
    the normalised :class:`SdkChannel` verbs. The function will be filled
    in once :func:`load_vendor_sdk` can succeed (i.e. once the SDK tarball
    is unpacked on the dev / cloud machine).

    Tests inject their own :class:`SdkChannel` directly, so the engine
    test suite does not depend on this factory being implemented.
    """
    _ = load_vendor_sdk()  # surfaces SdkLoadError early
    raise NotImplementedError(
        "Vendor SDK adaptor not implemented yet — pending SDK tarball "
        "unpacking on the cloud ECS (A2 Sprint 0 task 1.3). Tests use the "
        "InMemorySdkChannel from isales_engine.transport.aliyun_rtc."
    )
