"""In-process, typed, two-lane asyncio EventBus — one per CallSession.

Python port of voxen ``core/eventbus/eventbus.go``, adapted for iSales with the
ONE hard requirement voxen's uniform drop-oldest bus does not honour: a user's
ASR **final** (and interrupt / hangup / lifecycle) MUST NEVER be dropped — a
dropped final is a silently-missed customer turn. So delivery is split into two
lanes, chosen per event TYPE:

- **LOSSLESS** — unbounded queue, ``put_nowait`` physically cannot drop. Carries
  ``*Final`` / ``*Requested`` / ``*Detected`` / ``Connected`` / ``*Ended`` and
  anything not explicitly registered LOSSY (safe default — never silently drop).
- **LOSSY** — bounded, drop-oldest (voxen semantics). Carries ``*Partial`` /
  audio / VAD frames where loss is acceptable.

The single most important property carried over from the legacy
``run_loop._ListenContext`` coordination and HARDENED here is **per-handler
exception isolation**: today an unhandled exception in a pump kills the whole
pump (the 2026-05-31/06-05 "电话→AI 没通" class where one ``RtcError`` in
``bridge.py`` killed the entire uplink pump). Every handler is wrapped in
try/except so one bad handler can never take down the dispatch loop or sibling
handlers. ``CancelledError`` is always re-raised (barge-in self-cancel depends
on it).

Scope: purely intra-process, single-CallSession. NOT a Redis replacement — Redis
stays the inter-service boundary (see service-communication spec). One dispatcher
coroutine per lane: serialized within a lane (no CallSession-mutation races on
the lossless lane), concurrent across lanes (a partial/audio flood can never
delay a final).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
Handler = Callable[[Any], Awaitable[None] | None]

# Name-based default lane: an event type whose class name ends with one of these
# is LOSSLESS unless explicitly registered otherwise. Everything else also
# defaults LOSSLESS (safe — never silently drop); high-volume events that can
# tolerate loss MUST be registered LOSSY explicitly.
_LOSSLESS_DEFAULT_SUFFIXES = (
    "Final",
    "Requested",
    "Detected",
    "Connected",
    "Ended",
    "Started",
    "Done",
)


class Lane(Enum):
    LOSSLESS = "lossless"
    LOSSY = "lossy"


class EventBus:
    """A two-lane typed pub/sub for one call session.

    Usage::

        bus = EventBus()
        bus.register_lane(AsrPartial, Lane.LOSSY)
        unsub = bus.subscribe(AsrFinal, on_final)
        await bus.start()
        bus.post(AsrFinal(text="..."))   # non-blocking; lossless never drops
        ...
        await bus.aclose()
    """

    def __init__(self, *, lossy_maxsize: int = 256) -> None:
        self._lossless: asyncio.Queue[Any] = asyncio.Queue()
        self._lossy: asyncio.Queue[Any] = asyncio.Queue(maxsize=lossy_maxsize)
        self._lanes: dict[type, Lane] = {}
        self._handlers: dict[type, list[Handler]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False
        self.dropped = 0  # lossy-lane drop counter (observability)

    # ----- registration ---------------------------------------------------- #

    def register_lane(self, event_type: type, lane: Lane) -> None:
        """Override the default lane for *event_type*."""
        self._lanes[event_type] = lane

    def lane_for(self, event_type: type) -> Lane:
        lane = self._lanes.get(event_type)
        if lane is not None:
            return lane
        # Name-based default; unknown → LOSSLESS (safe: never silently drop).
        name = event_type.__name__
        if name.endswith(_LOSSLESS_DEFAULT_SUFFIXES):
            return Lane.LOSSLESS
        return Lane.LOSSLESS

    def subscribe(self, event_type: type[T], handler: Handler) -> Callable[[], None]:
        """Register *handler* for *event_type*. Returns an unsubscribe callable."""
        self._handlers.setdefault(event_type, []).append(handler)

        def _unsub() -> None:
            handlers = self._handlers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)

        return _unsub

    # ----- publish ---------------------------------------------------------- #

    def post(self, event: object) -> None:
        """Enqueue *event* on its lane. Non-blocking.

        Lossless: unbounded ``put_nowait`` — never drops. Lossy: drop-oldest on
        full (voxen semantics), bumping :attr:`dropped`.
        """
        if self._closed:
            return
        if self.lane_for(type(event)) is Lane.LOSSLESS:
            self._lossless.put_nowait(event)
            return
        try:
            self._lossy.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._lossy.get_nowait()  # drop oldest
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._lossy.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped += 1

    # ----- lifecycle -------------------------------------------------------- #

    async def start(self) -> None:
        """Spawn the two per-lane dispatcher coroutines."""
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(
                self._dispatch(self._lossless), name="eventbus_lossless"
            ),
            asyncio.create_task(self._dispatch(self._lossy), name="eventbus_lossy"),
        ]

    async def _dispatch(self, queue: asyncio.Queue[Any]) -> None:
        while True:
            event = await queue.get()
            try:
                await self._deliver(event)
            finally:
                queue.task_done()

    async def _deliver(self, event: object) -> None:
        # Snapshot the handler list so unsubscribe during delivery is safe.
        for handler in list(self._handlers.get(type(event), ())):
            try:
                result = handler(event)
                if result is not None:
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "eventbus_handler_error event=%s handler=%s",
                    type(event).__name__,
                    getattr(handler, "__qualname__", handler),
                )

    async def aclose(self) -> None:
        """Drain the lossless lane, then stop both dispatchers.

        Draining the lossless lane first preserves the shielded-finalize /
        DECR-snap-to-0 invariant — a final / CallEnded enqueued before close must
        still be delivered. Only events enqueued before ``aclose`` was called are
        drained (bounded — a handler that posts more lossless events during drain
        cannot loop forever).
        """
        if self._closed:
            return
        self._closed = True
        pending = self._lossless.qsize()
        for _ in range(pending):
            try:
                event = self._lossless.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await self._deliver(event)
            finally:
                self._lossless.task_done()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    # ----- test helper ------------------------------------------------------ #

    async def drain_inline(self) -> None:
        """Synchronously deliver all currently-queued events (test-mode).

        Lets event-sequence assertions stay deterministic without the background
        dispatcher tasks. Lossless first, then lossy — mirroring lane priority.
        """
        for queue in (self._lossless, self._lossy):
            while True:
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await self._deliver(event)
                finally:
                    queue.task_done()
