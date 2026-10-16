"""The lake sink: python -m services.lake_sink.main

Archives the stream's results in the lake. Reads `predictions` and `window-reports` and
writes them to object storage as Parquet, one folder per day:

    lake/outputs/predictions/date=2026-10-15/p0-000000001000-000000001999.parquet

It runs apart from the stream worker on purpose: the worker needs no storage library and
no storage credentials, and a slow or broken store never delays a hospital's scores. The
messages wait in the broker, and the sink catches up once the store is back.

Each file holds one fixed block of offsets of one partition, and is named after it. The
sink commits only up to the start of the first block it has not finished, so after a
crash it reads that block again and overwrites the same file with the same rows. The
blocks depend on the offsets alone, never on what one poll returned: that is what keeps a
re-run from writing a second, overlapping file. The block still filling is written too,
every `LAKE_FLUSH_S` seconds under its final name, so the newest results show up without
waiting for it to fill.

The worker delivers at least once, so one prediction can still appear twice, under two
offsets. Readers deduplicate by `event_id`.
"""

import io
import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from kafka import KafkaConsumer, TopicPartition
from kafka.structs import OffsetAndMetadata

log = logging.getLogger(__name__)

PREDICTIONS_TOPIC = "predictions"
WINDOWS_TOPIC = "window-reports"
PREFIX = "outputs"

# The columns the worker computes PSI for (stroke.drift.MONITORED; a test checks they
# match). Not imported from there: that module needs numpy and pandas, this image has neither.
PSI_COLUMNS = ("age", "avg_glucose_level", "bmi", "work_type", "smoking_status")

# Fixed schemas: only these columns are ever written. A field added to a message later
# does not reach the lake until it is added here, so patient features can't slip in.
PREDICTION_SCHEMA = pa.schema(
    [
        ("hospital_id", pa.string()),
        ("event_id", pa.string()),
        ("patient_id", pa.string()),  # the hospital's pseudonym, or null
        ("probability", pa.float64()),
        ("high_risk", pa.bool_()),
        ("threshold", pa.float64()),
        ("model_name", pa.string()),
        ("model_version", pa.string()),
        ("event_time", pa.float64()),
        ("scored_at", pa.float64()),
        ("kafka_offset", pa.int64()),
    ]
)
WINDOW_SCHEMA = pa.schema(
    [
        ("hospital_id", pa.string()),
        ("window_start", pa.float64()),
        ("window_end", pa.float64()),
        ("count", pa.int64()),
        ("throughput_eps", pa.float64()),
        ("latency_p50_ms", pa.float64()),
        ("latency_p95_ms", pa.float64()),
        ("positive_rate", pa.float64()),
        *[(f"psi_{col}", pa.float64()) for col in PSI_COLUMNS],
        ("warnings", pa.list_(pa.string())),
        ("alerts", pa.list_(pa.string())),
        ("model_versions", pa.list_(pa.string())),
        ("kafka_offset", pa.int64()),
    ]
)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def prediction_row(record: Any) -> dict:
    """A `predictions` message -> one row. The message has no hospital; its key is the hospital."""
    if record.key is None:
        raise ValueError("no key, so no hospital")
    v = json.loads(record.value)
    return {
        "hospital_id": record.key.decode(),
        "event_id": str(v["event_id"]),
        "patient_id": None if v.get("patient_id") is None else str(v["patient_id"]),
        "probability": float(v["probability"]),
        "high_risk": bool(v["high_risk"]),
        "threshold": float(v["threshold"]),
        "model_name": str(v["model_name"]),
        "model_version": str(v["model_version"]),
        "event_time": float(v["event_time"]),
        "scored_at": float(v["scored_at"]),
        "kafka_offset": record.offset,
    }


def window_row(record: Any) -> dict:
    """A `window-reports` message -> one row, with one flat column per PSI value."""
    v = json.loads(record.value)
    psi = v["psi"]
    return {
        "hospital_id": str(v["hospital_id"]),
        "window_start": float(v["window_start"]),
        "window_end": float(v["window_end"]),
        "count": int(v["count"]),
        "throughput_eps": _float_or_none(v["throughput_eps"]),
        "latency_p50_ms": _float_or_none(v["latency_p50_ms"]),
        "latency_p95_ms": _float_or_none(v["latency_p95_ms"]),
        "positive_rate": _float_or_none(v["positive_rate"]),
        **{f"psi_{col}": _float_or_none(psi.get(col)) for col in PSI_COLUMNS},
        "warnings": [str(col) for col in v["warnings"]],
        "alerts": [str(col) for col in v["alerts"]],
        "model_versions": [str(version) for version in v["model_versions"]],
        "kafka_offset": record.offset,
    }


@dataclass(frozen=True)
class Table:
    schema: pa.Schema
    row: Callable[[Any], dict]


TABLES = {
    PREDICTIONS_TOPIC: Table(PREDICTION_SCHEMA, prediction_row),
    WINDOWS_TOPIC: Table(WINDOW_SCHEMA, window_row),
}


def record_date(timestamp_ms: int) -> str:
    """The day a message was published, in UTC. It is stored with the message, so reading
    the message again, on any day, puts it in the same folder."""
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).date().isoformat()


def object_key(topic: str, partition: int, start: int, end: int, date: str) -> str:
    """Zero-padded offsets, so listing a folder lists its files in stream order."""
    return f"{PREFIX}/{topic}/date={date}/p{partition}-{start:012d}-{end:012d}.parquet"


def to_parquet(rows: list[dict], schema: pa.Schema) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), buf)
    return buf.getvalue()


class StoreError(RuntimeError):
    """A write failed. Nothing past it is committed; the sink reads those messages again."""


class S3Store:
    """Puts objects in one bucket. boto3 reads the endpoint and the credentials from the
    environment (AWS_ENDPOINT_URL_S3, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY): MinIO in
    compose, or AWS S3 itself when no endpoint is set. The code is the same for both."""

    def __init__(self, bucket: str, client: Any = None) -> None:
        self.bucket = bucket
        # Short timeouts: while a write hangs, the sink isn't reading, and a store that is
        # down should send it to its retry loop quickly.
        self.client = client or boto3.client(
            "s3",
            config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 3}),
        )

    def put(self, key: str, body: bytes) -> None:
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        except (BotoCoreError, ClientError) as exc:
            raise StoreError(f"s3://{self.bucket}/{key}: {exc}") from exc


@dataclass
class Block:
    """One partition's offsets [start, start + size): the rows read so far, by offset."""

    start: int
    size: int
    rows: dict[int, tuple[str, dict]] = field(default_factory=dict)  # offset -> (date, row)
    dirty: bool = False  # rows added since it was last written

    @property
    def end(self) -> int:
        return self.start + self.size - 1


class Sink:
    def __init__(
        self,
        consumer: Any,
        store: Any,
        block_records: int = 1000,
        flush_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.consumer, self.store = consumer, store
        self.block_records, self.flush_s, self.clock = block_records, flush_s, clock
        self.blocks: dict[TopicPartition, dict[int, Block]] = {}  # by start offset
        self.last_seen: dict[TopicPartition, int] = {}  # the highest offset read
        # Where a rewind goes back to: the last commit, else the first offset this run read.
        self.resume_at: dict[TopicPartition, int] = {}
        self.last_flush = clock()
        self.skipped = 0

    def add(self, tp: TopicPartition, records: list[Any]) -> None:
        for record in records:
            self.resume_at.setdefault(tp, record.offset)
            self.last_seen[tp] = max(self.last_seen.get(tp, -1), record.offset)
            start = record.offset - record.offset % self.block_records
            block = self.blocks.setdefault(tp, {}).setdefault(
                start, Block(start, self.block_records)
            )
            try:
                row = TABLES[tp.topic].row(record)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                # Skipped, not retried: the same bytes would fail the same way forever.
                self.skipped += 1
                log.warning("skipped %s/%d@%d: %r", tp.topic, tp.partition, record.offset, exc)
                continue
            # By offset, so a message read twice (after a rebalance) is stored once.
            block.rows[record.offset] = (record_date(record.timestamp), row)
            block.dirty = True

    def write(self, tp: TopicPartition, block: Block) -> None:
        """One file per day the block's rows fall on; usually one."""
        by_date: dict[str, list[dict]] = {}
        for offset in sorted(block.rows):
            date, row = block.rows[offset]
            by_date.setdefault(date, []).append(row)
        schema = TABLES[tp.topic].schema
        for date, rows in by_date.items():
            key = object_key(tp.topic, tp.partition, block.start, block.end, date)
            self.store.put(key, to_parquet(rows, schema))
        block.dirty = False

    def write_complete(self) -> None:
        """Write every block the stream has moved past, then commit up to the first block
        still filling. A block is complete once an offset at or past its end was read, so a
        gap in the offsets can't hold it open."""
        commits = {}
        for tp, blocks in self.blocks.items():
            for start in sorted(blocks):
                block = blocks[start]
                if self.last_seen[tp] < block.end:
                    break  # still filling, and so is every block after it
                if block.dirty:
                    self.write(tp, block)
                log.info(
                    "%s/%d offsets %d-%d: %d rows",
                    tp.topic,
                    tp.partition,
                    block.start,
                    block.end,
                    len(block.rows),
                )
                del blocks[start]
            commit_at = min(blocks) if blocks else self.last_seen[tp] + 1
            if commit_at > self.resume_at[tp]:
                commits[tp] = OffsetAndMetadata(commit_at, "", -1)
        if commits:
            self.consumer.commit(commits)
            self.resume_at.update({tp: om.offset for tp, om in commits.items()})

    def write_open(self) -> None:
        """Write the blocks still filling, under their final names, and commit nothing: the
        newest results become visible, and a crash still reads them again."""
        for tp, blocks in self.blocks.items():
            for block in blocks.values():
                if block.dirty:
                    self.write(tp, block)
        self.last_flush = self.clock()

    def process_once(self, max_records: int = 500) -> int:
        """One poll: add the records, write and commit the complete blocks, and write the
        open ones when the flush interval is up. Returns the records read."""
        batches = self.consumer.poll(timeout_ms=1000, max_records=max_records)
        for tp, records in batches.items():
            self.add(tp, records)
        self.write_complete()
        if self.clock() - self.last_flush >= self.flush_s:
            self.write_open()
        return sum(len(records) for records in batches.values())

    def rewind(self) -> None:
        """After a failed write: drop everything not committed and read it again. The rows
        are rebuilt from the broker, so none of them existed only in memory."""
        for tp, offset in self.resume_at.items():
            self.consumer.seek(tp, offset)
        self.blocks.clear()
        self.last_seen.clear()


def run(sink: Sink, stop: threading.Event, backoff_s: float = 5.0) -> None:
    while not stop.is_set():
        try:
            sink.process_once()
        except StoreError as exc:
            log.warning("write failed (%s); reading the uncommitted messages again", exc)
            sink.rewind()
            stop.wait(backoff_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    consumer = KafkaConsumer(
        PREDICTIONS_TOPIC,
        WINDOWS_TOPIC,
        bootstrap_servers=bootstrap,
        group_id="lake-sink",
        enable_auto_commit=False,  # committed by hand, block by block, after the write
        auto_offset_reset="earliest",  # a new group archives everything still retained
    )
    store = S3Store(os.getenv("LAKE_BUCKET", "lake"))
    sink = Sink(
        consumer,
        store,
        block_records=int(os.getenv("LAKE_BLOCK_RECORDS", "1000")),
        flush_s=float(os.getenv("LAKE_FLUSH_S", "30")),
    )

    stop = threading.Event()

    def shutdown(*_: object) -> None:
        log.info("shutting down after the current poll")
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info(
        "lake sink: %s -> %s/%s/%s/, files of %d offsets, open files written every %.0f s",
        bootstrap,
        store.client.meta.endpoint_url,
        store.bucket,
        PREFIX,
        sink.block_records,
        sink.flush_s,
    )
    try:
        run(sink, stop)
        sink.write_open()  # the newest rows; they are read again on restart anyway
    except StoreError as exc:
        log.warning("final write failed (%s); the next run writes those rows", exc)
    finally:
        consumer.close()  # auto-commit is off, so this commits nothing unwritten


if __name__ == "__main__":
    main()
