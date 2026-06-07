"""Golden-transcript regression net — change-0 safety net for the flat refactor.

This is the non-negotiable net that MUST stay green through every phase of the
EventBus / SelectRouter / flatten refactor (see
``isales/openspec/engine-flat-refactor-blueprint.md`` §5 change-0). It drives
``run_session()`` through representative scenarios with the deterministic mock
harness (``test_run_loop`` builders) and asserts ``session.full_transcript`` +
``session.pipeline_trace_records`` are **structurally identical** to committed
golden fixtures.

Timing-only fields (``ts``, ``*_ms``, wall-clock anchors) are canonicalised out
— the refactor is allowed to change *how fast* / *in what coordination shape*
the engine runs, but MUST NOT change *what conversation happens* (event types,
order, text, referee categories, matched rules, restructure flags, hangup
cause, state transitions). A diff here = the rewrite changed observable behavior.

Regenerate goldens **intentionally** (e.g. when change-3 deliberately changes
behavior) with::

    ISALES_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_transcript.py

then review the fixture diff under ``tests/golden/`` before committing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from isales_engine.providers.asr_mock import ScriptedMockASR
from isales_engine.realtime.mock_telephony import MockTelephonyClient
from isales_engine.run_loop import run_session

# The canonical deterministic mock harness lives in test_run_loop; reuse it so
# the net pins the SAME scenarios the smoke tests already prove deterministic.
from tests.test_run_loop import _make_config, _make_providers, _make_session

GOLDEN_DIR = Path(__file__).parent / "golden"

# Fields whose VALUE is non-deterministic (wall clock / elapsed time) and must
# be blanked before comparison. Anything ending in these suffixes is also
# blanked (covers nested referee_results[].duration_ms, *_at_monotonic, etc.).
_VOLATILE_KEYS = frozenset(
    {"ts", "ts_start", "ts_end", "first_audio_ms"}
)
_VOLATILE_SUFFIXES = ("_ms", "_at", "_at_monotonic", "_at_wallclock")
# engine-tools-multidialogue-gating: pipeline_trace gating columns are dropped
# from the golden net — their values are asserted in the dedicated gating /
# persona / tool tests (tests/test_gating.py); the golden net pins the observable
# *conversation* (transcript + main-reply trace), not the routing internals.
_GATING_KEYS = frozenset(
    {"selected_route_id", "selected_route_kind", "persona_candidates"}
)
_SENTINEL = "<volatile>"


def _canon(obj: Any) -> Any:
    """Recursively blank volatile timing fields; leave structure intact."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _GATING_KEYS:
                # DROP (not blank) flag-ON-only gating columns so the OFF fixture
                # (which lacks the keys) and the ON snapshot have identical key
                # sets for the cross-flag comparison.
                continue
            if k in _VOLATILE_KEYS or k.endswith(_VOLATILE_SUFFIXES):
                out[k] = _SENTINEL
            else:
                out[k] = _canon(v)
        return out
    if isinstance(obj, list):
        return [_canon(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Surface any non-JSON leftover (e.g. a stray datetime) deterministically
    # instead of crashing json.dumps — if this fires, add the key to _VOLATILE_*.
    return f"<nonjson:{type(obj).__name__}>"


def _snapshot(session: Any) -> dict[str, Any]:
    return {
        "transcript": _canon(session.full_transcript),
        "pipeline_trace": _canon(session.pipeline_trace_records),
    }


# --------------------------------------------------------------------------- #
# Scenarios — each returns a finished CallSession. Drivers are copied verbatim
# from the proven test_run_loop smoke tests (deterministic, feed-driven).
# --------------------------------------------------------------------------- #


async def _scenario_one_turn_hangup() -> Any:
    """Greeting → 1 user turn → AI reply → remote hangup."""
    session = _make_session()
    config = _make_config()
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("你好这是用户的回复内容")
        await asyncio.sleep(0.15)
        await tel.simulate_remote_hangup(1)

    task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await task
    return session


async def _scenario_goal_achieved_wrapup() -> Any:
    """Goal achieved (预约) → WRAPPING_UP → wrap-up round exhausts → hangup.

    Exercises the sales soft-outcome path (goal_achieved → wrap-up) that the
    flatten turns into a closing-persona route + then_state — the highest-value
    behavior to pin before the refactor.
    """
    session = _make_session()
    config = _make_config(wrap_up_max_rounds=1, silence_threshold_ms=3000)
    asr = ScriptedMockASR(partial_step_ms=5)
    providers = _make_providers(asr=asr)
    tel = MockTelephonyClient(connect_delay_ms=0)

    async def driver() -> None:
        await asyncio.sleep(0.05)
        await asr.feed_turn("我想预约一下")
        await asyncio.sleep(0.15)
        await asr.feed_turn("好的明白")
        await asyncio.sleep(0.15)

    task = asyncio.create_task(driver())
    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    await task
    return session


async def _scenario_silence_activation_hangup() -> Any:
    """No user speech → silence activation → still silent → silence hangup."""
    session = _make_session()
    config = _make_config(silence_threshold_ms=100)
    providers = _make_providers(asr=ScriptedMockASR(partial_step_ms=5))
    tel = MockTelephonyClient(connect_delay_ms=0)

    await run_session(
        session,
        phone="+8613800000000",
        config=config,
        telephony=tel,
        providers=providers,
    )
    return session


_SCENARIOS = {
    "one_turn_hangup": _scenario_one_turn_hangup,
    "goal_achieved_wrapup": _scenario_goal_achieved_wrapup,
    "silence_activation_hangup": _scenario_silence_activation_hangup,
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
async def test_golden_transcript(name: str) -> None:
    session = await _SCENARIOS[name]()
    snap = _snapshot(session)
    golden = GOLDEN_DIR / f"{name}.json"

    if os.environ.get("ISALES_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden.write_text(
            json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"regenerated golden fixture {golden.name}")

    assert golden.exists(), (
        f"missing golden fixture {golden}; regenerate with "
        f"ISALES_UPDATE_GOLDEN=1 python -m pytest {Path(__file__).name}"
    )
    expected = json.loads(golden.read_text(encoding="utf-8"))
    # Round-trip the live snapshot through JSON so the comparison is on the same
    # (str-keyed, tuple→list normalised) shape the fixture was written in.
    actual = json.loads(json.dumps(snap, ensure_ascii=False, sort_keys=True))
    assert actual == expected, (
        f"golden-transcript drift in scenario '{name}': the refactor changed "
        f"observable conversation behavior. If intentional, regenerate with "
        f"ISALES_UPDATE_GOLDEN=1 and review the fixture diff."
    )
