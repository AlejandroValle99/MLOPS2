"""The stream worker: python -m services.stream_worker.main

Reads patient-intake events from `patient-events` and checks each one against the
contract. Valid events are scored through the model server, one PredictBatch call per
poll, and each score is published to `predictions`, keyed by hospital, for that hospital
to read. Events that fail the check go to `patient-events-dlq` instead, so one bad event
never blocks the rest. Every scored event also feeds its hospital's drift window; a
sustained shift is published to `drift-alerts`.

Delivery is at least once: a batch's offsets are committed only after everything it
produced is on the broker, so a crash can repeat a few predictions but never lose one.
"""

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import grpc
from kafka import KafkaConsumer, KafkaProducer

from services.stream_worker.events import IntakeEvent, split_events
from services.stream_worker.windows import Monitor
from stroke_rpc.convert import request_from_record
from stroke_rpc.v1 import scoring_pb2 as pb
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

EVENTS_TOPIC = "patient-events"
PREDICTIONS_TOPIC = "predictions"
ALERTS_TOPIC = "drift-alerts"
DLQ_TOPIC = "patient-events-dlq"

# A dead-letter message keeps at most this much of the rejected event: enough to see what
# was wrong, and far below the broker's 1 MB message limit even after JSON escaping.
DLQ_RAW_CHARS = 4096

Scorer = Callable[[list[IntakeEvent]], list[pb.Prediction]]


def make_scorer(stub: pb_grpc.ScoringStub, timeout_s: float) -> Scorer:
    """Score a batch of events in ONE PredictBatch call; the results come back in order."""

    def score(events: list[IntakeEvent]) -> list[pb.Prediction]:
        request = pb.PredictBatchRequest(
            requests=[request_from_record(e.patient.to_model_row(), e.event_id) for e in events]
        )
        predictions = list(stub.PredictBatch(request, timeout=timeout_s))
        if len(predictions) != len(events):
            raise RuntimeError(f"sent {len(events)} events, got {len(predictions)} predictions")
        return predictions

    return score


def _json(value: dict) -> bytes:
    return json.dumps(value, allow_nan=False).encode()


def prediction_message(event: IntakeEvent, p: pb.Prediction, scored_at: float) -> bytes:
    """What the hospital receives. Its pseudonymous patient id, never the features."""
    return _json(
        {
            "event_id": event.event_id,
            "patient_id": event.patient.patient_id,
            "probability": p.probability,
            "high_risk": p.high_risk,
            "threshold": p.threshold,
            "model_name": p.model_name,
            "model_version": p.model_version,
            "event_time": event.event_time,
            "scored_at": scored_at,
        }
    )


def dead_letter_message(raw: bytes | None, error: str, rejected_at: float) -> bytes:
    """The rejected event as received (its first DLQ_RAW_CHARS characters), plus why. It
    holds features, so the topic keeps it only briefly (see docker-compose.yml).

    The cap matters: a large undecodable event embedded whole would grow about six-fold
    once JSON-escaped, the broker would refuse the dead-letter message, and the worker
    would stop on that same event after every restart."""
    text = raw.decode("utf-8", errors="replace") if raw is not None else None
    return _json(
        {
            "error": error,
            "raw": text[:DLQ_RAW_CHARS] if text is not None else None,
            "raw_bytes": len(raw) if raw is not None else 0,
            "truncated": text is not None and len(text) > DLQ_RAW_CHARS,
            "rejected_at": rejected_at,
        }
    )


def publish_report(producer: Any, report: dict) -> list[Any]:
    """Log a closed window; publish one alert per feature that fired. Returns the sends."""
    log.info("window %s", json.dumps(report))
    sent = []
    for feature in report["alerts"]:
        alert = {
            "hospital_id": report["hospital_id"],
            "feature": feature,
            "psi": report["psi"][feature],
            "window_start": report["window_start"],
            "window_end": report["window_end"],
            "model_versions": report["model_versions"],
        }
        log.warning("drift alert %s", json.dumps(alert))
        key = report["hospital_id"].encode()
        sent.append(producer.send(ALERTS_TOPIC, key=key, value=_json(alert)))
    return sent


def rewind(consumer: Any, batches: dict) -> None:
    """Move each partition back to the first offset of this batch. Not committing is not
    enough on its own: the consumer's position has already moved past these records, so
    without a seek the next poll would skip them and they would be lost."""
    for partition, records in batches.items():
        consumer.seek(partition, records[0].offset)


def process_once(
    consumer: Any,
    producer: Any,
    score: Scorer,
    monitor: Monitor,
    max_records: int = 256,
    clock: Callable[[], float] = time.time,
) -> int:
    """One poll: validate, score, publish, flush, commit. Returns the records handled.

    Raises grpc.RpcError, after rewinding, when the model server fails for any reason but
    refusing the events (down, slow, a crashing model, credentials): the caller waits and
    calls again, and the same events come back. Raises the
    broker's error, before committing, if it refused any message this batch produced: the
    worker stops, and on restart it resumes from the last commit, so nothing is lost.
    """
    batches = consumer.poll(timeout_ms=500, max_records=max_records)
    records = [record for partition_records in batches.values() for record in partition_records]
    if not records:
        return 0

    valid, rejected = split_events(records, monitor.baselines.keys())
    predictions: list[pb.Prediction] = []
    if valid:
        try:
            predictions = score([event for _, event in valid])
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.INVALID_ARGUMENT:
                # The service's fault, not the events': keep them and try again. Dead-lettering
                # here would silently drop every hospital's scores while the server is broken.
                rewind(consumer, batches)
                raise
            # The server refused events that passed our own check, so the two contracts
            # disagree. Retrying would fail the same way forever and block the partition:
            # dead-letter the batch instead, and keep the stream moving.
            reason = f"model server rejected the batch: {exc.code().name}: {exc.details()}"
            rejected += [(record, reason) for record, _ in valid]
            valid = []

    now = clock()
    sent = []  # send() is asynchronous: each call returns a future, checked after the flush
    for record, error in rejected:
        message = dead_letter_message(record.value, error, now)
        sent.append(producer.send(DLQ_TOPIC, key=record.key, value=message))
    for (_, event), p in zip(valid, predictions, strict=True):
        key = event.hospital_id.encode()
        message = prediction_message(event, p, now)
        sent.append(producer.send(PREDICTIONS_TOPIC, key=key, value=message))
        report = monitor.observe(
            event.hospital_id,
            event.patient.to_model_row(),
            latency_ms=(now - event.event_time) * 1000,
            high_risk=p.high_risk,
            model_version=p.model_version,
        )
        if report is not None:
            sent += publish_report(producer, report)

    # flush() returns once every send has finished, but finished includes refused: it does
    # not raise. get() on each future does, so a refused message stops us here, before the
    # commit, and its event is read again after the restart instead of being lost.
    producer.flush()
    for future in sent:
        future.get(timeout=10)  # already finished; raises if the broker refused it
    consumer.commit()  # only now are the offsets marked as done: at least once
    if rejected:
        log.warning("dead-lettered %d of %d events", len(rejected), len(records))
    return len(records)


def run(
    consumer: Any,
    producer: Any,
    score: Scorer,
    monitor: Monitor,
    stop: threading.Event,
    backoff_s: float = 2.0,
) -> None:
    while not stop.is_set():
        try:
            process_once(consumer, producer, score, monitor)
        except grpc.RpcError as exc:
            log.warning(
                "model server call failed (%s); retrying the same events in %.0f s",
                exc.code().name,
                backoff_s,
            )
            stop.wait(backoff_s)


def load_baselines(path: Path) -> dict[str, dict]:
    """{hospital_id: baseline}. The hospitals listed here are the ones the worker accepts."""
    return json.loads(path.read_text())["hospitals"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    baselines = load_baselines(
        Path(os.getenv("REFERENCE_STATS_PATH", "data/drift/reference_stats.json"))
    )
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    consumer = KafkaConsumer(
        EVENTS_TOPIC,
        bootstrap_servers=bootstrap,
        group_id="stream-worker",
        enable_auto_commit=False,  # committed by hand, after the results are out
        auto_offset_reset="earliest",  # a new group starts from the oldest retained event
    )
    producer = KafkaProducer(bootstrap_servers=bootstrap, acks="all", linger_ms=5)
    # No (de)serializers: keys and values stay bytes, so a malformed message reaches
    # split_events (and the dead-letter topic) instead of crashing inside poll().
    target = os.getenv("MODEL_SERVER_TARGET", "localhost:50051")
    channel = grpc.insecure_channel(target)
    score = make_scorer(pb_grpc.ScoringStub(channel), float(os.getenv("RPC_TIMEOUT_S", "2")))
    monitor = Monitor(baselines, window_events=int(os.getenv("WINDOW_EVENTS", "300")))

    stop = threading.Event()

    def shutdown(*_: object) -> None:
        log.info("shutting down after the current batch")
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info(
        "stream worker: %s -> %s, hospitals %s, windows of %d events",
        bootstrap,
        target,
        sorted(baselines),
        monitor.window_events,
    )
    try:
        run(consumer, producer, score, monitor, stop)
    finally:
        consumer.close()  # auto-commit is off, so this commits nothing unfinished
        producer.close()  # flushes anything still buffered
        channel.close()


if __name__ == "__main__":
    main()
