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

import asyncio
import importlib
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from isales_common.audio.rtc import RtcPushBackpressure

from isales_engine.transport._rtc_driver import (
    DriverQueueFull,
    _ArtcDriverThread,
)

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

    Lifecycle methods (:meth:`join`, :meth:`leave`, :meth:`push_audio`)
    are async because the production adaptor (:class:`_AliyunArtcChannel`)
    dispatches vendor SDK calls onto a dedicated driver thread (see
    :mod:`isales_engine.transport._rtc_driver`). Test doubles can
    implement them as trivial ``async def`` (no await) since their work
    is in-memory.

    Callback registration is sync; callbacks themselves fire on the
    driver thread (production) or the caller thread (in-memory test
    double). :class:`AliyunRtcSession` is responsible for marshaling
    those callbacks onto the engine event loop via
    ``loop.call_soon_threadsafe``.
    """

    async def join(
        self,
        channel: str,
        token: str,
        uid: str,
        *,
        send_sample_rate: int,
        send_channels: int,
    ) -> None:
        """Open a channel and join as ``uid`` with ``publisher_subscriber`` role.

        Resolves once the SDK reports the channel joined (vendor's
        ``OnJoinChannelResult`` callback fires). Raises on auth / network
        failure.
        """
        ...

    async def leave(self) -> None:
        """Leave the channel, release SDK resources. Idempotent."""
        ...

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        """Push outbound PCM. Returns 0 on success.

        Returns a positive int when the SDK's outbound buffer is full
        (vendor's ``ERR_AUDIO_BUFFER_FULL``); callers SHOULD await a
        :data:`BufferStateCallback` with ``is_full=False`` before retrying.

        MAY raise :class:`RtcPushBackpressure` when the driver-thread
        command queue is at its cap — that's a stronger backpressure
        signal than vendor's per-frame return code (queue saturation
        means the driver thread itself can't keep up).
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

    Threading model — the critical bit:

    Vendor's ``AliRTCEngineImpl`` is Python ↔ TCP-sidecar IPC. Inside
    ``CreateAliRTCEngine`` it registers two long-lived asyncio tasks
    (``__recvCoroutine`` + ``__heartbeatCoroutine``) on whatever loop is
    current; both only progress while their loop is being driven. Vendor
    SDK entry points (``JoinChannel``, ``Release``, …) then call
    ``loop.run_until_complete(__writeData(...))`` against the SAME loop.

    Therefore every vendor API call and every pump iteration must run on
    the SAME thread against the SAME loop. We achieve that by owning a
    :class:`_ArtcDriverThread` per channel instance and dispatching all
    vendor calls through ``self._driver.call(...)``. See
    :mod:`isales_engine.transport._rtc_driver` for the pump details.

    Vendor callback handling:

    - ``OnJoinChannelResult`` → unblocks :meth:`join` via an asyncio
      ``Event`` on the main loop (callback fires on the driver thread,
      so the ``Event`` set is marshaled with ``call_soon_threadsafe``).
    - ``OnSubscribeAudioFrame`` → :data:`InboundFrameCallback`.
    - ``OnPushAudioFrameBufferFull`` → :data:`BufferStateCallback`.
    - ``OnLeaveChannelResult`` / ``OnError`` / ``OnWarning`` → logged.

    Lifecycle: one instance per call. Reuse after :meth:`leave` is not
    supported (the vendor's engine instance is released, and the driver
    thread is joined).
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
        # Driver thread is spawned lazily in join() and torn down in
        # leave(); see design Decision 5 — __init__ may run before token
        # mint succeeds and we don't want a thread for a session that
        # never joins.
        self._driver: _ArtcDriverThread | None = None
        # OnJoinChannelResult is set on the driver thread; marshaled to
        # the main loop via call_soon_threadsafe so join() can await it.
        self._join_event: asyncio.Event | None = None
        self._join_result: int | None = None
        # Cached so the vendor-callback closure can post back to the
        # session's loop (callbacks fire on the driver thread).
        self._main_loop: asyncio.AbstractEventLoop | None = None

    # ----- SdkChannel surface --------------------------------------------

    def on_inbound_frame(self, callback: InboundFrameCallback) -> None:
        self._inbound_cb = callback

    def on_buffer_state(self, callback: BufferStateCallback) -> None:
        self._buffer_cb = callback

    async def join(
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

        self._main_loop = asyncio.get_running_loop()
        self._join_event = asyncio.Event()
        self._join_result = None
        self._handler = self._build_handler()

        # Spawn driver thread BEFORE first vendor call — both
        # CreateAliRTCEngine and JoinChannel must run on its loop.
        self._driver = _ArtcDriverThread(name=f"artc-driver-{uid}")
        self._driver.start(main_loop=self._main_loop)

        try:
            await self._driver.call(self._do_create_engine)
            await self._driver.call(
                self._do_configure_audio,
                send_sample_rate=send_sample_rate,
                send_channels=send_channels,
            )
            await self._driver.call(
                self._do_join_channel,
                channel=channel,
                token=token,
                uid=uid,
            )
            # Vendor's JoinChannel is fire-and-forget at the IPC layer;
            # OnJoinChannelResult fires on a later driver-thread pump
            # iteration. 30s timeout matches the previous behavior.
            try:
                await asyncio.wait_for(self._join_event.wait(), timeout=30.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "OnJoinChannelResult timed out after 30s",
                ) from exc
            if self._join_result != 0:
                raise RuntimeError(
                    f"OnJoinChannelResult reported failure: result={self._join_result}",
                )
        except BaseException:
            # join failed mid-flight — tear down the driver so we don't
            # leak a thread for a session that never went live. The
            # engine itself may or may not exist; best-effort release.
            await self._teardown_driver()
            raise

    async def leave(self) -> None:
        if self._engine is None and self._driver is None:
            return
        if self._engine is not None and self._driver is not None:
            try:
                await self._driver.call(self._do_leave_engine)
            except BaseException:  # noqa: BLE001
                # Even if the vendor cleanup misbehaves, we still need to
                # stop the driver thread — don't let a leaky release pin
                # the thread forever.
                logger.exception("error during AliyunArtcChannel.leave vendor cleanup")
        await self._teardown_driver()

    async def push_audio(self, pcm: bytes, *, timestamp_ms: int) -> int:
        if self._engine is None or self._driver is None:
            raise RuntimeError("push_audio() called before join()")
        try:
            return await self._driver.call(
                self._do_push_audio,
                pcm=pcm,
                timestamp_ms=timestamp_ms,
            )
        except DriverQueueFull as exc:
            # Translate driver-level saturation into the public RTC
            # backpressure error so callers handle it uniformly with the
            # Windows / macOS sessions.
            raise RtcPushBackpressure(
                "artc driver command queue full",
            ) from exc

    # ----- driver-thread closures (vendor SDK calls) --------------------

    def _do_create_engine(self) -> None:
        """Runs on the driver thread. Vendor's CreateAliRTCEngine
        internally does loop.run_until_complete(InitializeEngine) which
        registers the long-lived recvCoroutine on the driver loop."""
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

    def _do_configure_audio(
        self,
        *,
        send_sample_rate: int,
        send_channels: int,
    ) -> None:
        """Runs on the driver thread."""
        defs = self._defs
        self._engine.PublishLocalAudioStream(True)
        self._engine.SetExternalAudioSource(
            True,
            sampleRate=send_sample_rate,
            channelsPerFrame=send_channels,
        )
        self._engine.SetClientRole(
            defs.AliEngineClientRole.AliEngineClientRoleInteractive,
        )

    def _do_join_channel(
        self,
        *,
        channel: str,
        token: str,
        uid: str,
    ) -> None:
        """Runs on the driver thread. Vendor's JoinChannel internally
        does loop.run_until_complete(__writeData(...)); the driver thread
        owns the loop so this is safe."""
        defs = self._defs
        join_cfg = defs.JoinChannelConfig()
        join_cfg.channelProfile = defs.ChannelProfile.ChannelProfileInteractiveLive
        join_cfg.subscribeAudioFormat = defs.AudioFormat.AudioFormatPcmBeforMixing
        join_cfg.subscribeVideoFormat = defs.VideoFormat.VideoFormatH264
        join_cfg.isAudioOnly = True
        join_cfg.publishAvsyncMode = defs.PublishAvsyncMode.PublishAvsyncWithPts
        join_cfg.subscribeMode = defs.SubscribeMode.SubscribeAutomatically
        join_cfg.publishMode = defs.PublishMode.PublishAutomatically

        ret = self._engine.JoinChannel(token, channel, uid, uid, join_cfg)
        if ret != 0:
            raise RuntimeError(f"JoinChannel returned non-zero status: {ret}")

    def _do_leave_engine(self) -> None:
        """Runs on the driver thread."""
        if self._engine is None:
            return
        try:
            self._engine.PublishLocalAudioStream(False)
            self._engine.LeaveChannel()
            # Note: vendor recommends calling Release() AFTER leave
            # completes; we don't wait for OnLeaveChannelResult since the
            # caller is already moving on. Release is best-effort.
            self._engine.Release()
        finally:
            self._engine = None

    def _do_push_audio(self, *, pcm: bytes, timestamp_ms: int) -> int:
        """Runs on the driver thread."""
        return int(
            self._engine.PushExternalAudioFrameRawData(pcm, len(pcm), timestamp_ms),
        )

    async def _teardown_driver(self) -> None:
        """Stop the driver thread and join it; idempotent.

        Runs ``stop`` (which blocks until ``thread.join``) inside
        ``asyncio.to_thread`` so we don't pin the main loop for up to 5s.
        """
        driver = self._driver
        self._driver = None
        self._handler = None
        self._engine = None
        self._join_event = None
        self._join_result = None
        self._main_loop = None
        if driver is None:
            return
        await asyncio.to_thread(driver.stop)

    # ----- event handler subclass ----------------------------------------

    def _build_handler(self) -> Any:
        """Subclass the vendor's EngineEventHandlerInterface in-place.

        Vendor invokes these callbacks on the driver thread (inside the
        pump's ``run_until_complete``). Anything that touches asyncio
        state on the main loop MUST be marshaled with
        ``call_soon_threadsafe``.
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
                # _join_event is an asyncio.Event on the MAIN loop;
                # cannot call .set() directly from this driver thread.
                main_loop = outer._main_loop
                event = outer._join_event
                if main_loop is None or event is None:
                    return
                main_loop.call_soon_threadsafe(event.set)

            def OnLeaveChannelResult(self, result: int) -> None:
                logger.info("ARTC OnLeaveChannelResult result=%s", result)

            def OnSubscribeAudioFrame(self, uid: str, frame: Any) -> None:
                if outer._inbound_cb is None:
                    return
                pcm = frame.pcm
                if pcm is None or pcm.pcmBuf_ is None:
                    return
                # Vendor's AudioPcmFrame doesn't carry a wall-clock
                # timestamp; the engine session uses its own monotonic
                # clock for jitter buffer.
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
                logger.warning(
                    "ARTC OnError code=%s",
                    getattr(error_code, "value", error_code),
                )

            def OnWarning(self, warning_code: Any) -> None:
                logger.debug(
                    "ARTC OnWarning code=%s",
                    getattr(warning_code, "value", warning_code),
                )

        return Handler()
