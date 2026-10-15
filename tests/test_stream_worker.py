"""services.stream_worker.main: one poll from validation to commit, the retry path, and the
worker's slim imports. Scoring goes through a real local gRPC model server (the
`model_server` and `stub` fixtures in conftest.py); the broker is a pair of fakes that
behave like kafka-python's consumer and producer: a position that moves on poll and on
seek, commits of that position, and sends that only count once flushed."""

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest

from scripts.mini_tp1_client import VALID
from services.stream_worker.main import (
    ALERTS_TOPIC,
    DLQ_TOPIC,
    PREDICTIONS_TOPIC,
    WINDOWS_TOPIC,
    make_scorer,
    process_once,
    publish_report,
    run,
)
from services.stream_worker.windows import Monitor

ROOT = Path(__file__).parent.parent
BASELINES = json.loads((ROOT / "data/drift/reference_stats.json").read_text())["hospitals"]
PARTITION = ("patient-events", 0)
MAX_MESSAGE = 1_048_576  # kafka-python's default max_request_size


def event(n: int, hospital: str = "rural-60plus", **patient) -> bytes:
    body = {**VALID, "patient_id": f"p-{n}", **patient}
    message = {"hospital_id": hospital, "event_id": f"e-{n}", "event_time": 1.0, "patient": body}
    return json.dumps(message).encode()


def key_of(value: bytes) -> bytes | None:
    """The key the real producer sends: the event's hospital (none for non-events)."""
    try:
        return json.loads(value)["hospital_id"].encode()
    except (ValueError, KeyError, TypeError):
        return None


class FakeConsumer:
    """One partition. poll() hands out records from the current position and moves it;
    seek() moves it back; commit() records the position, as kafka-python's does."""

    def __init__(self, values: list[bytes], log: list):
        self.records = [
            SimpleNamespace(offset=i, key=key_of(v), value=v) for i, v in enumerate(values)
        ]
        self.position, self.committed, self.log = 0, None, log

    def poll(self, timeout_ms: int, max_records: int) -> dict:
        batch = self.records[self.position : self.position + max_records]
        self.position += len(batch)
        return {PARTITION: batch} if batch else {}

    def seek(self, partition, offset: int) -> None:
        self.log.append(("seek", offset))
        self.position = offset

    def commit(self) -> None:
        self.log.append(("commit", self.position))
        self.committed = self.position


class FakeFuture:
    """What send() returns. get() raises if the broker refused the message."""

    def __init__(self, error: Exception | None = None):
        self.error = error

    def get(self, timeout: float) -> None:
        if self.error is not None:
            raise self.error


class FakeProducer:
    def __init__(self, log: list):
        self.pending: list[tuple] = []
        self.sent: list[tuple] = []  # (topic, key, decoded value), once flushed
        self.refuse: set[str] = set()  # topics whose messages the "broker" refuses
        self.log = log

    def send(self, topic: str, key: bytes | None, value: bytes) -> FakeFuture:
        if topic in self.refuse:
            return FakeFuture(RuntimeError(f"broker refused a message to {topic}"))
        if len(value) > MAX_MESSAGE:
            return FakeFuture(RuntimeError(f"message too large ({len(value)} bytes)"))
        self.pending.append((topic, key, json.loads(value)))
        return FakeFuture()

    def flush(self) -> None:
        self.sent += self.pending
        self.pending = []
        self.log.append(("flush", len(self.sent)))

    def on(self, topic: str) -> list[tuple]:
        return [(key, value) for t, key, value in self.sent if t == topic]


@pytest.fixture
def broker():
    log: list = []
    producer = FakeProducer(log)
    return SimpleNamespace(log=log, producer=producer, consumer=lambda v: FakeConsumer(v, log))


@pytest.fixture
def score(stub):
    return make_scorer(stub, timeout_s=5)


@pytest.fixture
def monitor():
    return Monitor(BASELINES, window_events=300)


def test_valid_events_are_scored_and_published_to_their_hospital(
    broker, score, monitor, model_server, make_state
):
    model_server.holder.set(make_state(version="7"))
    consumer = broker.consumer([event(0), event(1, hospital="urban-60plus"), event(2)])
    assert process_once(consumer, broker.producer, score, monitor) == 3
    published = broker.producer.on(PREDICTIONS_TOPIC)
    assert [key for key, _ in published] == [b"rural-60plus", b"urban-60plus", b"rural-60plus"]
    assert [value["event_id"] for _, value in published] == ["e-0", "e-1", "e-2"]
    assert all(value["model_version"] == "7" for _, value in published)
    assert consumer.committed == 3


def test_what_the_hospital_receives_holds_no_patient_features(
    broker, score, monitor, model_server, make_state
):
    model_server.holder.set(make_state())
    process_once(broker.consumer([event(0)]), broker.producer, score, monitor)
    ((_, message),) = broker.producer.on(PREDICTIONS_TOPIC)
    assert message["patient_id"] == "p-0"
    assert set(message) == {
        "event_id", "patient_id", "probability", "high_risk", "threshold",
        "model_name", "model_version", "event_time", "scored_at",
    }  # fmt: skip


def test_offsets_are_committed_only_after_the_results_are_flushed(
    broker, score, monitor, model_server, make_state
):
    model_server.holder.set(make_state())
    process_once(broker.consumer([event(0), event(1)]), broker.producer, score, monitor)
    assert broker.log == [("flush", 2), ("commit", 2)]


def test_a_bad_event_is_dead_lettered_and_the_rest_of_the_batch_is_scored(
    broker, score, monitor, model_server, make_state
):
    model_server.holder.set(make_state())
    consumer = broker.consumer([event(0), event(1, age=-1), b"{not json", event(3)])
    process_once(consumer, broker.producer, score, monitor)
    assert [v["event_id"] for _, v in broker.producer.on(PREDICTIONS_TOPIC)] == ["e-0", "e-3"]
    dead = broker.producer.on(DLQ_TOPIC)
    assert len(dead) == 2
    assert "patient.age" in dead[0][1]["error"]
    assert dead[1][1]["raw"] == "{not json"  # kept as received, for whoever inspects it
    assert consumer.committed == 4  # the stream moves on


def test_a_batch_with_nothing_valid_is_committed_without_scoring(broker, monitor):
    def must_not_score(events):
        raise AssertionError("nothing valid should reach the model server")

    consumer = broker.consumer([b"[]", event(1, hospital="mars-base")])
    process_once(consumer, broker.producer, must_not_score, monitor)
    assert len(broker.producer.on(DLQ_TOPIC)) == 2
    assert consumer.committed == 2


def test_an_unavailable_model_server_rewinds_and_the_retry_scores_the_same_events(
    broker, score, monitor, model_server, make_state
):
    consumer = broker.consumer([event(0), event(1)])
    # No model loaded yet: the server answers UNAVAILABLE, which is worth retrying.
    with pytest.raises(grpc.RpcError) as failure:
        process_once(consumer, broker.producer, score, monitor)
    assert failure.value.code() == grpc.StatusCode.UNAVAILABLE
    assert broker.log == [("seek", 0)]  # back to the batch's first offset, nothing committed
    assert broker.producer.on(PREDICTIONS_TOPIC) == []

    model_server.holder.set(make_state())
    assert process_once(consumer, broker.producer, score, monitor) == 2  # the same two events
    assert [v["event_id"] for _, v in broker.producer.on(PREDICTIONS_TOPIC)] == ["e-0", "e-1"]
    assert consumer.committed == 2


def test_a_message_the_broker_refuses_stops_the_worker_before_the_commit(
    broker, score, monitor, model_server, make_state
):
    """flush() doesn't raise for a refused message; the worker must check each send, or it
    would commit an event whose prediction never reached the hospital."""
    model_server.holder.set(make_state())
    broker.producer.refuse = {PREDICTIONS_TOPIC}
    consumer = broker.consumer([event(0), event(1)])
    with pytest.raises(RuntimeError, match="refused"):
        process_once(consumer, broker.producer, score, monitor)
    assert consumer.committed is None  # read again after the restart, not lost


class ServerError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


def test_a_batch_the_model_server_refuses_is_dead_lettered_not_retried(broker, monitor):
    def refuse(events):
        raise ServerError(grpc.StatusCode.INVALID_ARGUMENT, "requests[0]: age must be in (0, 120]")

    consumer = broker.consumer([event(0), event(1)])
    process_once(consumer, broker.producer, refuse, monitor)
    dead = broker.producer.on(DLQ_TOPIC)
    assert len(dead) == 2 and "INVALID_ARGUMENT" in dead[0][1]["error"]
    assert consumer.committed == 2  # retrying would fail forever and block the partition


@pytest.mark.parametrize(
    "code", [grpc.StatusCode.INTERNAL, grpc.StatusCode.UNKNOWN, grpc.StatusCode.UNAUTHENTICATED]
)
def test_any_other_server_error_is_retried_not_dead_lettered(broker, monitor, code):
    """Only a contract disagreement is the events' fault. A crashing model or a credentials
    problem is the service's: dead-lettering would drop every hospital's scores."""

    def fail(events):
        raise ServerError(code, "something broke on the server")

    consumer = broker.consumer([event(0), event(1)])
    with pytest.raises(grpc.RpcError):
        process_once(consumer, broker.producer, fail, monitor)
    assert broker.log == [("seek", 0)]
    assert consumer.committed is None
    assert broker.producer.pending == [] and broker.producer.on(DLQ_TOPIC) == []


def test_an_oversized_garbage_event_is_dead_lettered_in_a_bounded_message(broker, monitor):
    """Embedding a huge undecodable event whole would make the dead-letter message itself too
    large for the broker: refused, never committed, and read again on every restart."""
    garbage = b"\xff" * 900_000  # each byte decodes to U+FFFD, six bytes once JSON-escaped
    consumer = broker.consumer([garbage])
    process_once(consumer, broker.producer, lambda events: [], monitor)
    ((_, dead),) = broker.producer.on(DLQ_TOPIC)
    assert dead["truncated"] is True and dead["raw_bytes"] == 900_000
    assert len(json.dumps(dead)) < MAX_MESSAGE
    assert consumer.committed == 1  # the stream moves on


def test_run_waits_and_retries_until_the_model_server_is_back(
    broker, score, monitor, model_server, make_state
):
    consumer = broker.consumer([event(0)])
    stop = threading.Event()
    calls = []

    def flaky_score(events):
        calls.append(len(events))
        if len(calls) == 1:
            return score(events)  # no model yet: UNAVAILABLE
        model_server.holder.set(make_state())
        stop.set()  # one successful pass is enough for this test
        return score(events)

    guard = threading.Timer(5, stop.set)  # if the retry never comes, fail instead of hanging
    guard.start()
    run(consumer, broker.producer, flaky_score, monitor, stop, backoff_s=0)
    guard.cancel()
    assert calls == [1, 1]
    assert consumer.committed == 1


def test_a_closed_window_with_an_alert_is_published_to_drift_alerts(broker):
    report = {
        "hospital_id": "rural-60plus",
        "window_start": 1.0,
        "window_end": 9.0,
        "psi": {"avg_glucose_level": 0.71, "age": 0.02},
        "alerts": ["avg_glucose_level"],
        "model_versions": ["7"],
    }
    publish_report(broker.producer, report)
    broker.producer.flush()
    ((key, alert),) = broker.producer.on(ALERTS_TOPIC)
    assert key == b"rural-60plus"
    assert alert["feature"] == "avg_glucose_level" and alert["psi"] == 0.71


def test_every_closed_window_is_published_to_window_reports_alert_or_not(broker):
    report = {
        "hospital_id": "urban-under60",
        "window_start": 1.0,
        "window_end": 9.0,
        "psi": {"avg_glucose_level": 0.03},
        "alerts": [],
        "model_versions": ["7"],
    }
    publish_report(broker.producer, report)
    broker.producer.flush()
    assert broker.producer.on(WINDOWS_TOPIC) == [(b"urban-under60", report)]
    assert broker.producer.on(ALERTS_TOPIC) == []


def test_a_window_report_holds_counts_and_psi_never_patient_features(
    broker, score, model_server, make_state
):
    model_server.holder.set(make_state(version="7"))
    consumer = broker.consumer([event(n) for n in range(3)])
    process_once(consumer, broker.producer, score, Monitor(BASELINES, window_events=3))
    ((key, report),) = broker.producer.on(WINDOWS_TOPIC)
    assert key == b"rural-60plus" and report["count"] == 3
    assert report["model_versions"] == ["7"]
    assert set(report) == {
        "hospital_id", "window_start", "window_end", "count", "throughput_eps",
        "latency_p50_ms", "latency_p95_ms", "positive_rate", "psi", "warnings", "alerts",
        "model_versions",
    }  # fmt: skip
    assert consumer.committed == 3  # the report was flushed with the predictions


def test_the_worker_imports_neither_scikit_learn_nor_mlflow():
    """Its image installs only the stream group: importing scikit-learn, mlflow, FastAPI or
    boto3 anywhere in the worker would crash it at startup in Docker."""
    code = (
        "import sys, services.stream_worker.main; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'sklearn', 'mlflow', 'fastapi', 'boto3', 'xgboost'}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
