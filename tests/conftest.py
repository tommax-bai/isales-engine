"""Shared pytest fixtures.

Real PG / Redis fixtures land alongside the components that need them
(session_manager / transcript_recorder in PR #2, dial consumer in PR #3,
end-to-end harness in PR #11). PR #1 keeps this minimal.
"""

from __future__ import annotations
