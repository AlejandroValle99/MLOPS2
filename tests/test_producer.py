"""services.producer.main: the replayed intake events, the drift switch and bad events."""

import json

import pytest

from services.producer.main import events, pseudonym
from services.stream_worker.events import parse
from stroke.data import HOSPITALS


def test_every_replayed_event_passes_the_workers_contract(dataset):
    """All 5,109 rows, serialized exactly as the producer sends them, are accepted: the
    replay alone never sends anything to the dead-letter topic."""
    X, _ = dataset
    stream = list(events(X))
    assert len(stream) == len(X)
    for e in stream:
        parse(json.dumps(e, allow_nan=False).encode(), HOSPITALS)  # raises if refused
    assert len({e["event_id"] for e in stream}) == len(stream)


def test_passes_repeat_the_same_patients_with_fresh_event_ids(dataset):
    X, _ = dataset
    first, second = [list(events(X.head(50), passes=2))[i::50] for i in (0, 1)]
    assert first[0]["patient"] == first[1]["patient"]
    assert first[0]["event_id"] != first[1]["event_id"]
    assert second[0]["patient"] == second[1]["patient"]


def test_the_drift_switch_shifts_one_column_at_one_hospital_from_one_event_on(dataset):
    X, _ = dataset
    plain = list(events(X, pseudonym_key=b"k"))
    shifted = list(
        events(X, pseudonym_key=b"k", drift_hospital="rural-60plus", drift_from=1000, factor=1.2)
    )
    for n, (a, b) in enumerate(zip(plain, shifted, strict=True)):
        if n >= 1000 and a["hospital_id"] == "rural-60plus":
            glucose = a["patient"]["avg_glucose_level"] * 1.2
            assert b["patient"]["avg_glucose_level"] == pytest.approx(glucose)
            assert {**b["patient"], "avg_glucose_level": None} == {
                **a["patient"],
                "avg_glucose_level": None,
            }
        else:
            assert a["patient"] == b["patient"]


def test_a_factor_that_breaks_the_contract_is_refused(dataset):
    X, _ = dataset
    with pytest.raises(ValueError, match="above the contract"):
        next(events(X, drift_hospital="rural-60plus", factor=2.0))


def test_bad_every_makes_exactly_every_nth_event_invalid(dataset):
    X, _ = dataset
    stream = list(events(X.head(100), bad_every=10))
    bad = [n for n, e in enumerate(stream) if e["patient"]["age"] < 0]
    assert bad == list(range(9, 100, 10))


def test_patient_ids_are_stable_for_one_key_and_unguessable_without_it():
    assert pseudonym(17, b"key") == pseudonym(17, b"key")
    assert pseudonym(17, b"key") != pseudonym(18, b"key")
    assert pseudonym(17, b"key") != pseudonym(17, b"another key")


def test_each_run_keys_its_pseudonyms_afresh(dataset):
    """With no key given, a random one is drawn, so knowing the scheme and the ~5,000 row
    numbers isn't enough to map an id back to its row."""
    X, _ = dataset
    first, second = next(events(X)), next(events(X))
    assert first["patient"]["patient_id"] != second["patient"]["patient_id"]


class CountingBroker:
    """Stands in for KafkaProducer: refuses every 50th message, accepts the rest."""

    def __init__(self, **config):
        self.calls = 0

    def send(self, topic, key, value):
        from kafka.future import Future

        self.calls += 1
        future = Future()
        if self.calls % 50 == 0:
            return future.failure(RuntimeError("broker refused it"))
        return future.success(None)

    def flush(self):
        pass

    def close(self):
        pass


def test_main_reports_the_events_the_broker_refused(raw_csv, tmp_path, monkeypatch, caplog):
    small = tmp_path / "small.csv"
    small.write_text("".join(raw_csv.read_text().splitlines(keepends=True)[:201]))  # 200 rows
    monkeypatch.setattr("services.producer.main.KafkaProducer", CountingBroker)
    caplog.set_level("INFO")
    from services.producer.main import main

    main(["--csv", str(small), "--passes", "1", "--rate", "1000000"])
    assert "200 sent, 196 acknowledged, 4 refused" in caplog.text


def test_main_refuses_a_bad_factor_before_connecting_to_the_broker(monkeypatch):
    def no_broker(**config):
        raise AssertionError("the factor must be checked before connecting")

    monkeypatch.setattr("services.producer.main.KafkaProducer", no_broker)
    from services.producer.main import main

    with pytest.raises(SystemExit, match="factor"):
        main(["--drift-hospital", "rural-60plus", "--factor", "2.0", "--passes", "1"])
