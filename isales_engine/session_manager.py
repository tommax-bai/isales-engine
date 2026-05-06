"""Process-wide registry of in-flight :class:`CallSession` objects.

PR #2 ships the registry semantics (register / unregister / cancel_all). The
SIGTERM-driven graceful shutdown (which calls ``cancel_all`` and waits for
finally blocks to flush) lands in PR #13.
"""

from __future__ import annotations

import asyncio
import logging

from isales_engine.call_session import CallSession

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[int, CallSession] = {}

    def register(self, session: CallSession) -> None:
        if session.call_record_id in self._sessions:
            logger.warning(
                "session_already_registered call_record_id=%s", session.call_record_id
            )
            return
        self._sessions[session.call_record_id] = session

    def unregister(self, call_record_id: int) -> CallSession | None:
        return self._sessions.pop(call_record_id, None)

    def get(self, call_record_id: int) -> CallSession | None:
        return self._sessions.get(call_record_id)

    def active_count(self) -> int:
        return len(self._sessions)

    def all_sessions(self) -> list[CallSession]:
        return list(self._sessions.values())

    async def cancel_all(self, *, timeout_s: float = 30.0) -> None:
        """Cancel every session's owned tasks, waiting up to ``timeout_s`` seconds.

        Each session's ``run()`` is expected to consume the ``CancelledError``
        in its finally block (writing call_record / LPUSH CallEnded / DECR
        concurrency) — so this method waits for those finalisers, not just the
        cancellation acknowledgement.
        """

        if not self._sessions:
            return

        all_tasks: list[asyncio.Task[object]] = []
        for session in self._sessions.values():
            for task in session.tasks.values():
                task.cancel()
                all_tasks.append(task)

        if not all_tasks:
            return

        try:
            async with asyncio.timeout(timeout_s):
                await asyncio.gather(*all_tasks, return_exceptions=True)
        except TimeoutError:
            logger.warning(
                "session_manager_shutdown_timeout active=%s timeout=%s",
                len(self._sessions),
                timeout_s,
            )
