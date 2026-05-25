"""Daemon thread + periodic-pump asyncio loop for the vendor ARTC SDK.

The Aliyun ARTC SDK for Linux Python is a Python ↔ TCP-sidecar IPC model.
Inside :meth:`AliRTCEngineImpl.__InitializeEngine` (vendor) it registers
two long-lived asyncio tasks on whatever loop is current:

- ``__recvCoroutine`` — reads sidecar TCP responses; dispatches into
  ``OnJoinChannelResult`` / ``OnSubscribeAudioFrame`` / etc. callbacks.
- ``__heartbeatCoroutine`` — keep-alive ping to the sidecar.

Both only progress when their loop is being driven. Vendor's own
``demo.py`` drives them with a ``while not done: sleepFor(0.1)`` loop,
where ``sleepFor`` calls ``loop.run_until_complete(asyncio.sleep(0.1))``.

Vendor SDK API entry points (``JoinChannel`` at
``AliRTCEngineImpl.py:1135-1169``, ``Release`` at ``:1066-1081``, etc.)
themselves call ``loop.run_until_complete(__writeData(...))`` inside a
``with self.__lock:`` block. So:

1. The loop must NOT be permanently running (``run_until_complete`` would
   raise ``RuntimeError: This event loop is already running``).
2. Every vendor API call AND every pump iteration must run on the SAME
   thread, against the SAME loop, because they all use
   ``asyncio.get_event_loop()`` internally.

This module owns the thread and loop; :class:`_AliyunArtcChannel` posts
vendor API calls to it via :meth:`_ArtcDriverThread.call` and pumps in
between via :meth:`_ArtcDriverThread._run`'s pump iteration.

Spec: device-hardware § Requirement: 云端 isales-engine
``transport/aliyun_rtc.py`` 通过专用 SDK 驱动线程承载 vendor recvCoroutine.

Root-cause history: ``project_artc_join_diagnosis_2026_05_25.md`` —
without this thread, ECS smoke saw 0 HTTP requests to
``gw.rtn.aliyuncs.com`` because the vendor's recvCoroutine never got CPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["_ArtcDriverThread", "_DEFAULT_PUMP_INTERVAL_S", "_PUMP_INTERVAL_ENV"]

logger = logging.getLogger(__name__)


_PUMP_INTERVAL_ENV = "ISALES_ARTC_PUMP_INTERVAL_MS"
_DEFAULT_PUMP_INTERVAL_S = 0.100  # 100ms — vendor demo's `sleepFor(0.1)` rhythm.
# Cap intentionally low: queue overflow signals real backpressure (push_audio
# faster than the driver thread can drain), not transient blips. 256 ≈ 5s
# worth of 50ms-frame TTS pushes — past that, the call's likely wedged.
_DEFAULT_CMD_QUEUE_CAP = 256


def _read_pump_interval() -> float:
    raw = os.environ.get(_PUMP_INTERVAL_ENV)
    if raw is None:
        return _DEFAULT_PUMP_INTERVAL_S
    try:
        ms = float(raw)
    except ValueError:
        logger.warning(
            "ignoring non-numeric %s=%r; falling back to %sms",
            _PUMP_INTERVAL_ENV,
            raw,
            int(_DEFAULT_PUMP_INTERVAL_S * 1000),
        )
        return _DEFAULT_PUMP_INTERVAL_S
    if ms <= 0:
        logger.warning(
            "ignoring non-positive %s=%r; falling back to %sms",
            _PUMP_INTERVAL_ENV,
            raw,
            int(_DEFAULT_PUMP_INTERVAL_S * 1000),
        )
        return _DEFAULT_PUMP_INTERVAL_S
    return ms / 1000.0


class DriverQueueFull(RuntimeError):
    """Raised by :meth:`_ArtcDriverThread.submit` when the command queue
    is at its configured cap.

    Higher layers (``AliyunRtcSession.push_audio``) translate this into
    :class:`isales_common.audio.rtc.RtcPushBackpressure` so callers see
    the standard backpressure error type.
    """


class _ArtcDriverThread:
    """A daemon thread that owns one asyncio loop for one ARTC engine.

    Lifecycle:

    1. ``__init__`` — does NOT start the thread; caller controls timing.
    2. ``start(main_loop=...)`` — spawns thread, creates loop, waits until
       the thread reports the loop is set on its thread-local.
    3. ``call(fn, *args, **kwargs)`` (coroutine) — submit a sync callable
       to run on the driver thread; await the main-loop ``Future`` it
       returns.
    4. ``submit(fn)`` — non-blocking enqueue; returns the main-loop
       ``Future`` directly (used by ``push_audio`` fast path that doesn't
       want to ``await`` per-frame).
    5. ``stop(timeout=...)`` — signal stop, push a wakeup sentinel,
       ``thread.join(timeout)``. Idempotent.

    Concurrency rules:

    - All vendor SDK API calls and the pump itself run on this thread —
      vendor's internal ``loop.run_until_complete(__writeData(...))`` thus
      always finds a loop that's not already running.
    - Vendor callbacks (``OnJoinChannelResult`` etc.) fire on this thread
      too (inside the pump's ``run_until_complete``). They marshal back
      to the engine session's loop via ``call_soon_threadsafe`` in
      :class:`_AliyunArtcChannel`'s event-handler subclass.
    - The driver thread itself never imports or touches asyncio objects
      from the main loop except via ``call_soon_threadsafe``.
    """

    def __init__(
        self,
        *,
        name: str = "artc-driver",
        pump_interval_s: float | None = None,
        cmd_queue_cap: int = _DEFAULT_CMD_QUEUE_CAP,
    ) -> None:
        self._name = name
        self._pump_interval_s = (
            pump_interval_s if pump_interval_s is not None else _read_pump_interval()
        )
        self._cmd_q: queue.Queue[
            tuple[Callable[[], Any], asyncio.Future[Any]] | None
        ] = queue.Queue(maxsize=cmd_queue_cap)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._healthy = True

    # ----- public API ----------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """``False`` once an unhandled exception escaped the pump loop.

        Engine sessions SHOULD observe this between iterations and trigger
        a clean leave + ``RtcError`` upcall when it flips. The thread is
        kept alive past the failure so in-flight commands fail loudly
        rather than silently hanging on their futures.
        """
        return self._healthy

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def queue_size(self) -> int:
        """Approximate command queue depth.

        Used by ``push_audio`` to decide whether to enqueue or raise
        backpressure. ``queue.Queue.qsize()`` is documented as approximate
        on multi-threaded Python, but here only this thread writes and
        only the driver thread reads, so it's tight enough for a soft cap.
        """
        return self._cmd_q.qsize()

    def start(self, *, main_loop: asyncio.AbstractEventLoop) -> None:
        """Spawn the driver thread; block (up to 5s) until its loop is set.

        ``main_loop`` is the asyncio loop where caller futures resolve.
        Stored once; calling :meth:`start` twice raises.
        """
        if self._thread is not None:
            raise RuntimeError("driver thread already started")
        self._main_loop = main_loop
        self._thread = threading.Thread(
            target=self._run,
            name=self._name,
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            # Driver thread didn't even reach asyncio.set_event_loop —
            # something is very wrong; tear down before propagating so we
            # don't leak the thread.
            self._stop.set()
            raise RuntimeError("artc driver thread failed to initialize within 5s")

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal stop and join the thread; idempotent.

        Logs WARN on join timeout but does not raise — leaving a single
        daemon thread is preferable to bubbling an exception out of
        ``leave`` / ``__del__`` paths.
        """
        if self._thread is None:
            return
        self._stop.set()
        # Wakeup sentinel: nudges the queue drain so we don't have to wait
        # a full pump interval to notice the stop flag. ``put_nowait`` is
        # safe even if the queue is at cap — at worst we lose one slot's
        # worth of capacity for the time it takes to drain.
        try:
            self._cmd_q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "artc driver thread %s did not stop within %.1fs; leaking thread",
                self._name,
                timeout,
            )
        self._thread = None

    def submit(self, fn: Callable[[], Any]) -> asyncio.Future[Any]:
        """Enqueue ``fn`` for execution on the driver thread; return its
        main-loop ``Future``.

        Raises :class:`DriverQueueFull` if the queue is at cap. Raises
        ``RuntimeError`` if the thread was never started or has already
        stopped.
        """
        if self._main_loop is None or self._thread is None:
            raise RuntimeError("driver thread not started")
        if not self._thread.is_alive():
            raise RuntimeError(
                f"artc driver thread {self._name} is not alive (healthy={self._healthy})",
            )
        fut: asyncio.Future[Any] = self._main_loop.create_future()
        try:
            self._cmd_q.put_nowait((fn, fut))
        except queue.Full as exc:
            raise DriverQueueFull(
                f"artc driver queue full (cap={self._cmd_q.maxsize})",
            ) from exc
        return fut

    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Submit ``fn(*args, **kwargs)`` and ``await`` its result.

        The submitted callable runs SYNCHRONOUSLY on the driver thread —
        if it internally does ``loop.run_until_complete(...)`` (which the
        vendor SDK does for every API entry point), that nested call uses
        the driver thread's loop.

        Exceptions raised by ``fn`` propagate out of the ``await``.
        """
        return await self.submit(lambda: fn(*args, **kwargs))

    # ----- driver-thread internals --------------------------------------

    def _run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        except BaseException:  # noqa: BLE001 — must not let thread die silently
            logger.exception(
                "artc driver thread %s failed during loop setup",
                self._name,
            )
            self._healthy = False
            # Still set _ready so start() unblocks and raises cleanly via
            # the timeout check; without this it would just hang 5s.
            self._ready.set()
            return
        self._ready.set()

        try:
            while not self._stop.is_set():
                self._drain_commands_once()
                # Pump: give vendor's __recvCoroutine + __heartbeatCoroutine
                # a chance to read sidecar TCP. Wrapped in its own try so a
                # single pump exception (e.g. vendor's recvCoroutine raises
                # because the sidecar died) doesn't kill the driver thread
                # — leaving in-flight cmds with futures that never resolve
                # would be the worst failure mode.
                try:
                    self._loop.run_until_complete(
                        asyncio.sleep(self._pump_interval_s),
                    )
                except BaseException:  # noqa: BLE001
                    logger.exception(
                        "artc driver pump iteration crashed (healthy → False)",
                    )
                    self._healthy = False
        except BaseException:  # noqa: BLE001
            logger.exception(
                "artc driver thread %s crashed out of main loop",
                self._name,
            )
            self._healthy = False
        finally:
            # Hard stop: any cmds still in the queue (or queued after the
            # stop flag flipped) fail with a clear error rather than
            # running to completion or dangling. Normal shutdown should
            # have drained the queue before calling stop() — anything
            # left at this point is racing with leave() and SHOULD fail
            # loudly so callers notice.
            self._fail_pending_commands(
                RuntimeError("artc driver thread stopping"),
            )
            try:
                self._loop.close()
            except BaseException:  # noqa: BLE001
                logger.exception(
                    "artc driver thread %s failed to close loop", self._name,
                )

    def _drain_commands_once(self) -> None:
        """Drain whatever's currently in the queue, non-blocking.

        Each command runs on this (driver) thread. Its result/exception
        is shipped back to the main loop's Future via
        ``call_soon_threadsafe``.

        Once ``self._stop`` flips (during shutdown), remaining queued
        cmds are failed with a clear ``RuntimeError`` rather than
        executed — the in-flight cmd that flipped the flag mid-execution
        is allowed to complete normally so its caller gets the result it
        was waiting on.
        """
        while True:
            try:
                item = self._cmd_q.get_nowait()
            except queue.Empty:
                return
            if item is None:
                # Wakeup sentinel from stop(); nothing to execute.
                continue
            fn, fut = item
            if self._stop.is_set():
                # Shutdown in progress — fail rather than execute so
                # callers don't hang on cmds that may never resolve.
                self._dispatch_future_exception(
                    fut, RuntimeError("artc driver thread stopping"),
                )
                continue
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001
                self._dispatch_future_exception(fut, exc)
            else:
                self._dispatch_future_result(fut, result)

    def _fail_pending_commands(self, exc: BaseException) -> None:
        """On thread exit, settle every still-queued command with ``exc``
        so awaiting callers don't hang on dangling futures."""
        while True:
            try:
                item = self._cmd_q.get_nowait()
            except queue.Empty:
                return
            if item is None:
                continue
            _, fut = item
            self._dispatch_future_exception(fut, exc)

    def _dispatch_future_result(
        self,
        fut: asyncio.Future[Any],
        result: Any,
    ) -> None:
        assert self._main_loop is not None  # narrowed by start() invariant
        self._main_loop.call_soon_threadsafe(_safe_set_result, fut, result)

    def _dispatch_future_exception(
        self,
        fut: asyncio.Future[Any],
        exc: BaseException,
    ) -> None:
        assert self._main_loop is not None  # narrowed by start() invariant
        self._main_loop.call_soon_threadsafe(_safe_set_exception, fut, exc)


def _safe_set_result(fut: asyncio.Future[Any], result: Any) -> None:
    """``Future.set_result`` is unsafe if the future was already cancelled
    by the caller (e.g. their task was cancelled mid-await). Guard it."""
    if not fut.done():
        fut.set_result(result)


def _safe_set_exception(fut: asyncio.Future[Any], exc: BaseException) -> None:
    if not fut.done():
        fut.set_exception(exc)
