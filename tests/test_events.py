"""services.stream_worker.events: the intake-event contract and the check before scoring."""

import json
from types import SimpleNamespace

import pytest

from scripts.mini_tp1_client import VALID
from services.stream_worker.events import IntakeEvent, parse, split_events

HOSPITALS = {"rural-60plus", "urban-60plus"}
EVENT = {"hospital_id": "rural-60plus", "event_id": "e-1", "event_time": 1.7e9, "patient": VALID}


def raw(event: dict) -> bytes:
    return json.dumps(event).encode()


def test_a_valid_event_parses_with_the_rest_contract_inside():
    event = parse(raw(EVENT), HOSPITALS)
    assert isinstance(event, IntakeEvent)
    assert event.patient.to_model_row()["Residence_type"] == VALID["residence_type"]


@pytest.mark.parametrize(
    "message, reason",
    [
        (b"{not json", "Invalid JSON"),
        (None, "empty message"),
        (raw({**EVENT, "patient": {**VALID, "age": -1}}), "patient.age"),
        (raw({**EVENT, "patient": {**VALID, "Residence_type": "Urban"}}), "patient.Residence_type"),
        (raw({k: v for k, v in EVENT.items() if k != "event_id"}), "event_id"),
        (raw({**EVENT, "hospital_id": "mars-base"}), "unknown hospital"),
        (raw({**EVENT, "extra": 1}), "extra"),
    ],
    ids=["not-json", "empty", "bad-age", "old-field", "no-event-id", "unknown-hospital", "extra"],
)
def test_a_bad_event_is_refused_with_a_reason_naming_the_problem(message, reason):
    with pytest.raises(ValueError, match=reason):
        parse(message, HOSPITALS)


def test_split_keeps_each_record_with_its_event_or_its_reason():
    good = SimpleNamespace(key=b"rural-60plus", value=raw(EVENT))
    bad = SimpleNamespace(key=None, value=b"[]")
    valid, rejected = split_events([good, bad, good], HOSPITALS)
    assert [record for record, _ in valid] == [good, good]
    assert [record for record, _ in rejected] == [bad]
    assert rejected[0][1]  # a non-empty reason


def test_an_event_whose_key_is_another_hospital_is_refused():
    """The worker routes each score by hospital_id; a key that disagrees would put one
    hospital's patients in another's partition, out of order with its own events."""
    valid, rejected = split_events(
        [
            SimpleNamespace(key=b"rural-60plus", value=raw(EVENT)),  # matches
            SimpleNamespace(key=None, value=raw(EVENT)),  # unkeyed: still routed correctly
            SimpleNamespace(key=b"urban-60plus", value=raw(EVENT)),  # mismatch
        ],
        HOSPITALS,
    )
    assert len(valid) == 2
    assert "does not match" in rejected[0][1]
