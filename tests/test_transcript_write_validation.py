"""Transcript write-time validation (engine-transcript-write-validation).

CallSession.append_event validates every event against the TranscriptEvent
contract: fail-fast under strict mode (CI), fail-soft + loud-log in production,
so engine-vs-schema drift can't silently persist and 500 GET /calls on read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from isales_common.schemas.jsonb import TranscriptEvent
from pydantic import TypeAdapter, ValidationError

from isales_engine.call_session import CallSession

_ADAPTER: TypeAdapter[object] = TypeAdapter(TranscriptEvent)
_GOLDEN_DIR = Path(__file__).parent / "golden"


def _session() -> CallSession:
    return CallSession(
        call_record_id=1,
        campaign_id=1,
        lead_id=1,
        caller_id="+8613900000000",
        prompt_versions_snapshot={},
    )


def test_valid_event_appends_without_error() -> None:
    session = _session()
    event = session.append_event("hangup", reason="user_hangup", initiated_by="user")
    assert event in session.full_transcript
    assert event["reason"] == "user_hangup"


def test_out_of_contract_event_raises_under_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISALES_ENGINE_STRICT_TRANSCRIPT", "1")
    session = _session()
    with pytest.raises(ValidationError):
        # rounds_exhausted is not in Literal["max_rounds", "max_seconds"]
        session.append_event("wrap_up_completed", reason="rounds_exhausted")
    # raise happens before the append — nothing persisted.
    assert not any(e["type"] == "wrap_up_completed" for e in session.full_transcript)


def test_out_of_contract_event_failsoft_under_non_strict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("ISALES_ENGINE_STRICT_TRANSCRIPT", raising=False)
    session = _session()
    with caplog.at_level("ERROR"):
        event = session.append_event("wrap_up_completed", reason="rounds_exhausted")
    # logged loudly...
    assert any(
        "transcript_event_schema_violation" in rec.getMessage()
        for rec in caplog.records
    )
    # ...but still persisted, and the call is not interrupted.
    assert event in session.full_transcript


def test_all_golden_transcripts_validate() -> None:
    goldens = sorted(_GOLDEN_DIR.glob("*.json"))
    assert goldens, "no golden transcripts found"
    for path in goldens:
        data = json.loads(path.read_text())
        events = data["transcript"]
        for event in events:
            # golden files use "<volatile>" for the wall-clock-relative ts.
            if event.get("ts") == "<volatile>":
                event = {**event, "ts": 0}
            _ADAPTER.validate_python(event)
