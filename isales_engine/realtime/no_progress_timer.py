"""No-progress timer.

Spec: call-state-machine § Scenario "长时间无进展挂断".

The user speaks repeatedly but every utterance is ignored as
non-interruption / invalid. After ``max_no_progress_seconds`` we hang up
with reason ``no_progress_timeout``.

This module is a pure helper — it just compares elapsed time. PR #11
maintains the underlying timestamp on :class:`CallSession`.
"""

from __future__ import annotations


def is_no_progress_exceeded(
    *,
    last_progress_ts_ms: int,
    now_ts_ms: int,
    max_no_progress_seconds: int | None,
) -> bool:
    if max_no_progress_seconds is None or max_no_progress_seconds <= 0:
        return False
    return (now_ts_ms - last_progress_ts_ms) >= max_no_progress_seconds * 1000
