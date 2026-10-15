"""services.lake_sink.main: which offsets go in which file, when the sink commits, how a
re-run or a failed write ends, and what a file holds. The broker and the store are fakes:
a consumer whose positions move on poll and on seek and that keeps explicit commits, and a
store that keeps objects in a dict. The messages are built by the stream worker's own
code, so the two sides of each topic are tested against each other."""

import io
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import boto3
import pyarrow.parquet as pq
import pytest
from botocore.stub import Stubber
from kafka import TopicPartition

from api.schemas import StrokeInput
from scripts.mini_tp1_client import VALID
from services.lake_sink import main as sink_main
from services.lake_sink.main import (
    PREDICTION_SCHEMA,
    PSI_COLUMNS,
    WINDOW_SCHEMA,
    S3Store,
    Sink,
    StoreError,
    run,
)
from services.stream_worker import main as worker_main
from services.stream_worker.events import IntakeEvent
from services.stream_worker.windows import Monitor
from stroke.drift import MONITORED
from stroke_rpc.v1 import scoring_pb2 as pb

ROOT = Path(__file__).parent.parent
BASELINES = json.loads((ROOT / "data/drift/reference_stats.json").read_text())["hospitals"]
P = TopicPartition("predictions", 0)
W = TopicPartition("window-reports", 0)


def ms(*when: int) -> int:
    return int(datetime(*when, tzinfo=UTC).timestamp() * 1000)


NOON = ms(2026, 10, 15, 12)


def prediction(n: int, hospital: str = "rural-60plus") -> bytes:
    """A `predictions` message exactly as the worker publishes it."""
    event = IntakeEvent.model_validate(
        {
            "hospital_id": hospital,
            "event_id": f"e-{n}",
            "event_time": 1.0,
            "patient": {**VALID, "patient_id": f"p-{n}"},
        }
    )
    p = pb.Prediction(
        probability=0.3, high_risk=True, threshold=0.1, model_name="stroke-model", model_version="6"
    )
    return worker_main.prediction_message(event, p, scored_at=2.0)


def record(tp, offset, value, ts=NOON, key=b"rural-60plus"):
    return SimpleNamespace(
        topic=tp.topic, partition=tp.partition, offset=offset, timestamp=ts, key=key, value=value
    )


def predictions(offsets, **kwargs) -> list:
    return [record(P, n, prediction(n), **kwargs) for n in offsets]


class FakeConsumer:
    """poll() hands out each partition's records from its position, at most `max_records`
    per call, and moves the position; seek() moves it back; commit() keeps the offsets."""

    def __init__(self, logs: dict, start: dict | None = None, max_records: int = 500):
        self.logs, self.max_records = logs, max_records
        self.position = {tp: (start or {}).get(tp, 0) for tp in logs}
        self.committed: dict = {}
        self.seeks: list = []

    def poll(self, timeout_ms: int, max_records: int) -> dict:
        out, budget = {}, min(max_records, self.max_records)
        for tp, records in self.logs.items():
            batch = [r for r in records if r.offset >= self.position[tp]][:budget]
            if batch:
                out[tp] = batch
                self.position[tp] = batch[-1].offset + 1
                budget -= len(batch)
        return out

    def seek(self, tp, offset: int) -> None:
        self.seeks.append((tp, offset))
        self.position[tp] = offset

    def commit(self, offsets: dict) -> None:
        self.committed.update({tp: om.offset for tp, om in offsets.items()})


class FakeStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.fail = False

    def put(self, key: str, body: bytes) -> None:
        if self.fail:
            raise StoreError("store down")
        self.objects[key] = body


def read(body: bytes) -> list[dict]:
    return pq.read_table(io.BytesIO(body)).to_pylist()


def drain(sink: Sink) -> None:
    while sink.process_once():
        pass


def key(n0: int, n1: int, date: str = "2026-10-15", topic: str = "predictions") -> str:
    return f"outputs/{topic}/date={date}/p0-{n0:012d}-{n1:012d}.parquet"


@pytest.fixture
def store():
    return FakeStore()


def test_a_full_block_is_written_once_and_the_commit_moves_to_the_next_block(store):
    consumer = FakeConsumer({P: predictions(range(6))})
    Sink(consumer, store, block_records=4, flush_s=60, clock=lambda: 0.0).process_once()
    assert list(store.objects) == [key(0, 3)]
    assert [row["kafka_offset"] for row in read(store.objects[key(0, 3)])] == [0, 1, 2, 3]
    assert consumer.committed == {P: 4}  # offsets 4 and 5 are read again after a restart


def test_a_rerun_after_a_crash_overwrites_the_same_files_with_the_same_bytes(store):
    logs = {P: predictions(range(10))}
    drain(Sink(FakeConsumer(logs), store, block_records=4, flush_s=0))
    first = dict(store.objects)
    assert sorted(first) == [key(0, 3), key(4, 7), key(8, 11)]  # 8-11 is still filling

    # The commits never reached the broker, and this run's polls split the stream
    # differently. The blocks, and so the files, are the same.
    drain(Sink(FakeConsumer(logs, max_records=3), store, block_records=4, flush_s=0))
    assert store.objects == first


def test_the_open_block_is_written_on_the_interval_without_committing_past_it(store):
    now = [0.0]
    logs = {P: predictions(range(5))}
    consumer = FakeConsumer(logs)
    sink = Sink(consumer, store, block_records=1000, flush_s=30, clock=lambda: now[0])
    sink.process_once()
    assert store.objects == {}  # the interval isn't up yet

    now[0] = 31.0
    sink.process_once()
    assert len(read(store.objects[key(0, 999)])) == 5
    assert consumer.committed == {}  # a crash now reads all five again

    logs[P] += predictions(range(5, 8))
    now[0] = 62.0
    sink.process_once()
    assert list(store.objects) == [key(0, 999)]  # the same file, now with eight rows
    assert len(read(store.objects[key(0, 999)])) == 8


def test_a_block_closes_once_the_stream_moves_past_it_even_across_offset_gaps(store):
    consumer = FakeConsumer({P: predictions([0, 1, 2, 5, 6])})  # 3 and 4 never arrive
    Sink(consumer, store, block_records=4, flush_s=60, clock=lambda: 0.0).process_once()
    assert [row["kafka_offset"] for row in read(store.objects[key(0, 3)])] == [0, 1, 2]
    assert consumer.committed == {P: 4}


def test_a_block_that_spans_midnight_is_written_as_one_file_per_day(store):
    late, early = ms(2026, 10, 15, 23, 59), ms(2026, 10, 16, 0, 1)
    logs = {P: predictions([0, 1], ts=late) + predictions([2, 3], ts=early)}
    Sink(FakeConsumer(logs), store, block_records=4, flush_s=60, clock=lambda: 0.0).process_once()
    assert sorted(store.objects) == [key(0, 3, "2026-10-15"), key(0, 3, "2026-10-16")]
    assert len(read(store.objects[key(0, 3, "2026-10-16")])) == 2


def test_a_prediction_file_holds_the_scores_and_no_patient_features(store):
    smuggled = json.loads(prediction(1))
    smuggled["age"] = 67  # a field the message gains later must not reach the lake
    logs = {P: [record(P, 0, prediction(0)), record(P, 1, json.dumps(smuggled).encode())]}
    Sink(FakeConsumer(logs), store, block_records=2, flush_s=60, clock=lambda: 0.0).process_once()

    table = pq.read_table(io.BytesIO(store.objects[key(0, 1)]))
    assert table.schema.equals(PREDICTION_SCHEMA)
    features = set(StrokeInput.model_fields) - {"patient_id"}
    assert features.isdisjoint(table.column_names)
    first = table.to_pylist()[0]
    assert first["hospital_id"] == "rural-60plus"  # from the message key
    assert (first["event_id"], first["patient_id"], first["model_version"]) == ("e-0", "p-0", "6")
    assert first["high_risk"] is True and first["probability"] == pytest.approx(0.3)


def test_window_reports_become_flat_rows_with_one_psi_column_per_feature(store):
    monitor = Monitor(BASELINES, window_events=3)
    row = StrokeInput(**VALID).to_model_row()
    for _ in range(3):
        report = monitor.observe("rural-60plus", row, 12.0, high_risk=True, model_version="6")
    logs = {W: [record(W, 0, json.dumps(report).encode())]}
    Sink(FakeConsumer(logs), store, block_records=1, flush_s=60, clock=lambda: 0.0).process_once()

    table = pq.read_table(io.BytesIO(store.objects[key(0, 0, topic="window-reports")]))
    assert table.schema.equals(WINDOW_SCHEMA)
    (written,) = table.to_pylist()
    assert written["count"] == 3 and written["positive_rate"] == 1.0
    assert written["model_versions"] == ["6"]
    assert all(written[f"psi_{col}"] == report["psi"][col] for col in PSI_COLUMNS)


def test_a_malformed_message_is_skipped_and_its_block_still_closes(store):
    logs = {
        P: [
            record(P, 0, prediction(0)),
            record(P, 1, b"not json"),
            record(P, 2, prediction(2), key=None),  # no hospital
            record(P, 3, prediction(3)),
        ]
    }
    consumer = FakeConsumer(logs)
    sink = Sink(consumer, store, block_records=4, flush_s=60, clock=lambda: 0.0)
    sink.process_once()
    assert [row["kafka_offset"] for row in read(store.objects[key(0, 3)])] == [0, 3]
    assert sink.skipped == 2 and consumer.committed == {P: 4}


def test_a_failed_write_commits_nothing_and_the_same_messages_are_read_again(store):
    consumer = FakeConsumer({P: predictions(range(4))})
    sink = Sink(consumer, store, block_records=4, flush_s=60, clock=lambda: 0.0)
    store.fail = True
    with pytest.raises(StoreError):
        sink.process_once()
    sink.rewind()
    assert consumer.committed == {} and consumer.seeks == [(P, 0)]

    store.fail = False
    sink.process_once()
    assert len(read(store.objects[key(0, 3)])) == 4
    assert consumer.committed == {P: 4}


def test_run_waits_and_retries_until_the_store_is_back():
    stop = threading.Event()

    class FlakyStore(FakeStore):
        calls = 0

        def put(self, key: str, body: bytes) -> None:
            self.calls += 1
            if self.calls == 1:
                raise StoreError("store down")
            super().put(key, body)
            stop.set()  # one successful write is enough for this test

    store = FlakyStore()
    consumer = FakeConsumer({P: predictions(range(4))})
    guard = threading.Timer(5, stop.set)  # if the retry never comes, fail instead of hanging
    guard.start()
    run(Sink(consumer, store, block_records=4, flush_s=60), stop, backoff_s=0)
    guard.cancel()
    assert store.calls == 2 and list(store.objects) == [key(0, 3)]
    assert consumer.committed == {P: 4}


def test_the_s3_store_puts_the_object_and_turns_a_refusal_into_a_store_error():
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )
    with Stubber(client) as stubber:
        stubber.add_response("put_object", {}, {"Bucket": "lake", "Key": "k", "Body": b"x"})
        S3Store("lake", client).put("k", b"x")
        stubber.add_client_error("put_object", "NoSuchBucket", http_status_code=404)
        with pytest.raises(StoreError, match="s3://missing/k: .*NoSuchBucket"):
            S3Store("missing", client).put("k", b"x")


def test_the_sink_reads_the_topics_and_psi_columns_the_worker_writes():
    assert sink_main.PREDICTIONS_TOPIC == worker_main.PREDICTIONS_TOPIC
    assert sink_main.WINDOWS_TOPIC == worker_main.WINDOWS_TOPIC
    assert list(PSI_COLUMNS) == MONITORED


def test_the_sink_imports_nothing_outside_its_own_group():
    """Its image installs only the lake group: kafka-python, boto3 and pyarrow."""
    code = (
        "import sys, services.lake_sink.main; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'sklearn', 'mlflow', 'fastapi', 'grpc', 'pydantic', 'xgboost', 'stroke'}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
