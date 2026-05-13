"""Vendor SDK boundary for the cloud-side RtcSession implementation.

The Aliyun ARTC SDK for Linux Python ships as a binary archive (download
URL: ``https://alivc-demo-cms.alicdn.com/versionProduct/sdk/linux/`` ―
authoritative source page:
https://help.aliyun.com/zh/live/artc-download-the-sdk). Once unpacked, the
SDK directory contains:

- ``Python/AliRTCEngine.py`` — high-level wrapper module (the entry point
  for ``CreateAliRTCEngine``).
- ``Python/AliRTCEngineImpl.py`` — internal implementation; we never touch
  it directly.
- ``Python/AliRTCLinuxSdkDefine.py`` — enums and value classes
  (``AuthInfo``, ``JoinChannelConfig``, ``AudioPcmFrame``, etc).
- ``Python/Release/lib/AliRtcCoreService`` — the native sidecar binary the
  Python wrapper drives via IPC (Linux x86_64 only).
- ``Python/Release/lib/libAliRtcLinuxEngine.so`` — Python wrapper's ``.so``.
- ``Python/Release/lib/libonnxruntime.so.1.16.3`` — ONNX runtime dependency.

The wrapper modules are pure Python and import cleanly on macOS for dev
inspection; only :func:`AliRTCEngine.CreateAliRTCEngine` actually loads the
native binaries (Linux-only). This lets the engine test suite exercise the
adaptor logic via mocks on any dev machine while still running unmodified
on the production ECS.

To keep the engine code testable and to absorb minor SDK API drift in one
place, we narrow the vendor surface to a small :class:`SdkChannel`
protocol with five methods. Concrete adaptors:

- :class:`_AliyunArtcChannel` (production): thin wrapper around the
  vendor's ``CreateAliRTCEngine`` + ``EngineEventHandlerInterface``
  subclass; loaded lazily so importing this module never touches the
  vendor wrapper.
- :class:`isales_engine.transport.aliyun_rtc.InMemorySdkChannel` (tests):
  loopback implementation that drives callbacks inline.

Spec: device-hardware § Requirement: 云端 engine 的 ARTC SDK 接入.

Vendor location: set ``ISALES_RTC_SDK_PATH`` to the absolute path of the
unpacked ``Python/`` directory. Default is the deploy-script convention
``/opt/isales/current/vendor/aliyun-artc-linux-python/Python``.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
    """Raised when the vendor SDK modules cannot be imported.

    Production deploy script (``deploy/cloud/scripts/install-artc-sdk.sh``)
    is responsible for unpacking the vendor tarball on the cloud ECS and
    setting ``ISALES_RTC_SDK_PATH`` to the extracted ``Python/`` directory.
    Tests pass a pre-built ``SdkChannel`` instance directly to
    :class:`AliyunRtcSession`, bypassing this loader.
    """


_DEFAULT_SDK_PATH_ENV = "ISALES_RTC_SDK_PATH"
_DEFAULT_SDK_PATH = "/opt/isales/current/vendor/aliyun-artc-linux-python/Python"

# The two Python modules we need from the vendor package. Names are fixed
# by the SDK distribution and have been stable across recent vendor releases.
_SDK_ENGINE_MODULE = "AliRTCEngine"
_SDK_DEFINE_MODULE = "AliRTCLinuxSdkDefine"


def _resolve_sdk_path() -> str:
    return os.environ.get(_DEFAULT_SDK_PATH_ENV, _DEFAULT_SDK_PATH)


def load_vendor_modules() -> tuple[Any, Any]:
    """Import the vendor SDK's two Python modules.

    Inserts ``ISALES_RTC_SDK_PATH`` (default
    ``/opt/isales/current/vendor/aliyun-artc-linux-python/Python``) into
    ``sys.path`` if it's not already present, then imports ``AliRTCEngine``
    and ``AliRTCLinuxSdkDefine``.

    Returns the ``(engine_module, define_module)`` tuple.

    Raises :class:`SdkLoadError` if either module is not importable.
    """
    sdk_path = _resolve_sdk_path()
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    try:
        engine_mod = importlib.import_module(_SDK_ENGINE_MODULE)
        define_mod = importlib.import_module(_SDK_DEFINE_MODULE)
    except ImportError as exc:
        raise SdkLoadError(
            f"Vendor SDK modules not importable from {sdk_path}. "
            "Run deploy/cloud/scripts/install-artc-sdk.sh on the cloud ECS "
            f"or set ${_DEFAULT_SDK_PATH_ENV} to the absolute path of the "
            "unpacked SDK's `Python/` directory."
        ) from exc
    return engine_mod, define_mod


def vendor_channel_factory(app_id: str) -> SdkChannel:
    """Construct a production :class:`SdkChannel` backed by the vendor SDK.

    The factory only validates that the vendor's Python wrapper modules
    import. The actual ``CreateAliRTCEngine`` call (which requires the
    Linux native ``.so`` + ``AliRtcCoreService`` binary) happens inside
    :meth:`_AliyunArtcChannel.join`, so this factory is safe to call on
    any dev machine to type-check the wiring.
    """
    engine_mod, define_mod = load_vendor_modules()
    return _AliyunArtcChannel(
        app_id=app_id,
        engine_module=engine_mod,
        define_module=define_mod,
    )


# =============================================================================
# Production adaptor: _AliyunArtcChannel
# =============================================================================


class _AliyunArtcChannel:
    """Production :class:`SdkChannel` over Aliyun ARTC SDK for Linux Python.

    Subclasses the vendor's :class:`EngineEventHandlerInterface` to route
    the relevant callbacks (out of ~70 total) back into the ``SdkChannel``
    contract:

    - ``OnJoinChannelResult`` → unblocks :meth:`join`.
    - ``OnSubscribeAudioFrame`` → :data:`InboundFrameCallback`.
    - ``OnPushAudioFrameBufferFull`` → :data:`BufferStateCallback`.
    - ``OnLeaveChannelResult`` → logged, not surfaced (caller already moved on).
    - ``OnError`` → logged.

    The vendor's ``JoinChannel`` is fire-and-forget; this adaptor blocks
    in :meth:`join` until ``OnJoinChannelResult`` fires, via a
    ``threading.Event``. :class:`AliyunRtcSession` calls ``join`` inside
    ``asyncio.to_thread`` so the asyncio event loop stays unblocked.

    Lifecycle: one instance per call. Reuse after :meth:`leave` is not
    supported (the vendor's engine instance is released).
    """

    def __init__(
        self,
        *,
        app_id: str,
        engine_module: Any,
        define_module: Any,
    ) -> None:
        self._app_id = app_id
        self._artc = engine_module
        self._defs = define_module
        self._engine: Any = None
        self._handler: Any = None
        self._inbound_cb: InboundFrameCallback | None = None
        self._buffer_cb: BufferStateCallback | None = None
        # OnJoinChannelResult signaling.
        self._join_event = threading.Event()
        self._join_result: int | None = None

    # ----- SdkChannel surface --------------------------------------------

    def on_inbound_frame(self, callback: InboundFrameCallback) -> None:
        self._inbound_cb = callback

    def on_buffer_state(self, callback: BufferStateCallback) -> None:
        self._buffer_cb = callback

    def join(
        self,
        channel: str,
        token: str,
        uid: str,
        *,
        send_sample_rate: int,
        send_channels: int,
    ) -> None:
        if self._engine is not None:
            raise RuntimeError("channel already joined")

        self._handler = self._build_handler()

        core_service_path = os.environ.get(
            "ISALES_RTC_CORE_SERVICE",
            os.path.join(_resolve_sdk_path(), "Release", "lib", "AliRtcCoreService"),
        )
        port_min = int(os.environ.get("ISALES_RTC_PORT_MIN", "42000"))
        port_max = int(os.environ.get("ISALES_RTC_PORT_MAX", "45000"))
        work_dir = os.environ.get("ISALES_RTC_WORK_DIR", "/tmp")
        # Disable the AI-vendor ranking module — iSales does its own
        # multi-role PK pipeline on top of plain PCM.
        extra = json.dumps({"user_specified_disable_audio_ranking": "true"})

        self._engine = self._artc.CreateAliRTCEngine(
            self._handler,
            port_min,
            port_max,
            work_dir,
            core_service_path,
            False,  # h5mode: iSales talks to a Linux SDK peer, not a browser.
            extra,
        )

        defs = self._defs
        join_cfg = defs.JoinChannelConfig()
        join_cfg.channelProfile = defs.ChannelProfile.ChannelProfileInteractiveLive
        join_cfg.subscribeAudioFormat = defs.AudioFormat.AudioFormatPcmBeforMixing
        join_cfg.subscribeVideoFormat = defs.VideoFormat.VideoFormatH264
        join_cfg.isAudioOnly = True
        join_cfg.publishAvsyncMode = defs.PublishAvsyncMode.PublishAvsyncWithPts
        join_cfg.subscribeMode = defs.SubscribeMode.SubscribeAutomatically
        join_cfg.publishMode = defs.PublishMode.PublishAutomatically

        self._engine.PublishLocalAudioStream(True)
        self._engine.SetExternalAudioSource(
            True,
            sampleRate=send_sample_rate,
            channelsPerFrame=send_channels,
        )
        self._engine.SetClientRole(
            defs.AliEngineClientRole.AliEngineClientRoleInteractive,
        )

        ret = self._engine.JoinChannel(token, channel, uid, uid, join_cfg)
        if ret != 0:
            raise RuntimeError(f"JoinChannel returned non-zero status: {ret}")

        # Wait for OnJoinChannelResult — vendor JoinChannel is asynchronous;
        # 30s is generous (typical join is sub-second).
        if not self._join_event.wait(timeout=30.0):
            raise RuntimeError("OnJoinChannelResult timed out after 30s")
        if self._join_result != 0:
            raise RuntimeError(
                f"OnJoinChannelResult reported failure: result={self._join_result}",
            )

    def leave(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.PublishLocalAudioStream(False)
            self._engine.LeaveChannel()
            # Note: vendor recommends calling Release() AFTER leave
            # completes; we don't wait for OnLeaveChannelResult since the
            # caller is already moving on. Release is best-effort.
            self._engine.Release()
        except Exception:  # noqa: BLE001 — vendor exceptions are unspecified
            logger.exception("error during AliyunArtcChannel.leave")
        finally:
            self._engine = None
            self._handler = None
            self._join_event.clear()
            self._join_result = None

    def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        if self._engine is None:
            raise RuntimeError("push_audio() called before join()")
        return int(
            self._engine.PushExternalAudioFrameRawData(pcm, len(pcm), timestamp_ms),
        )

    # ----- event handler subclass ----------------------------------------

    def _build_handler(self) -> Any:
        """Subclass the vendor's EngineEventHandlerInterface in-place.

        We need ``self`` from the enclosing :class:`_AliyunArtcChannel`
        instance to fire callbacks, so we build the subclass inside this
        method and capture ``outer = self`` in the closure.
        """
        outer = self
        Base = self._artc.EngineEventHandlerInterface  # noqa: N806 (vendor name)

        class Handler(Base):  # type: ignore[valid-type, misc]
            def OnJoinChannelResult(
                self,
                result: int,
                channel: str,
                userId: str,
            ) -> None:
                outer._join_result = result
                outer._join_event.set()

            def OnLeaveChannelResult(self, result: int) -> None:
                logger.info("ARTC OnLeaveChannelResult result=%s", result)

            def OnSubscribeAudioFrame(self, uid: str, frame: Any) -> None:
                if outer._inbound_cb is None:
                    return
                pcm = frame.pcm
                if pcm is None or pcm.pcmBuf_ is None:
                    return
                # Vendor's AudioPcmFrame doesn't carry a wall-clock
                # timestamp; surface frame_ms_ * sequence number is the
                # caller's concern. We pass 0 for timestamp_ms — the engine
                # session uses its own monotonic clock for jitter buffer.
                outer._inbound_cb(
                    uid,
                    bytes(pcm.pcmBuf_),
                    int(pcm.sample_rates_),
                    0,
                )

            def OnPushAudioFrameBufferFull(self, isFull: bool) -> None:  # noqa: N803
                if outer._buffer_cb is not None:
                    outer._buffer_cb(isFull)

            def OnError(self, error_code: Any) -> None:
                logger.warning("ARTC OnError code=%s", getattr(error_code, "value", error_code))

            def OnWarning(self, warning_code: Any) -> None:
                logger.debug("ARTC OnWarning code=%s", getattr(warning_code, "value", warning_code))

        return Handler()
