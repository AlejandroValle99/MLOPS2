"""Replays the dataset as patient-intake events, standing in for the hospitals' systems.

    uv run --group ml --group stream python -m services.producer.main --rate 200
    # glucose x1.2 at one hospital (a lab switching units, say), starting at the 3,000th
    # event overall, all hospitals counted (30 s in at the default 100 events/s):
    uv run --group ml --group stream python -m services.producer.main \\
        --drift-hospital rural-60plus --drift-from 3000 --factor 1.2
    # every 500th event invalid, to watch the worker dead-letter it:
    uv run --group ml --group stream python -m services.producer.main --bad-every 500

Each event is what a hospital's system would publish when a patient is registered: the
REST request body for that patient under a pseudonymous id, plus who sent it and when.
It runs from the laptop rather than in compose, because reading the dataset needs
scikit-learn (stroke.data), which the stream images don't have.
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

from stroke.data import HOSPITALS, assign_hospital, load_clean

log = logging.getLogger(__name__)

TOPIC = "patient-events"
RAW_CSV = Path(__file__).resolve().parents[2] / "data/raw/healthcare-dataset-stroke-data.csv"
GLUCOSE_LIMIT = 500  # the REST contract's upper bound for avg_glucose_level


def request_body(row: dict, patient_id: str) -> dict:
    """A dataset row -> the body the REST API (and the stream worker) accepts."""
    body = {k: v for k, v in row.items() if k != "Residence_type"}
    body["residence_type"] = row["Residence_type"]
    if pd.isna(body["bmi"]):
        body["bmi"] = None  # NaN is not valid JSON
    body["patient_id"] = patient_id
    return body


def pseudonym(row_index: object, key: bytes) -> str:
    """A stable id for one patient that doesn't reveal which dataset row it is. It is keyed
    (HMAC), as a hospital would key its own ids with a secret it keeps: without the key,
    trying all ~5,000 row numbers doesn't reverse it. The event still carries the patient's
    features, so this is pseudonymisation, not anonymisation."""
    return "p-" + hmac.new(key, str(row_index).encode(), hashlib.sha256).hexdigest()[:16]


def check_factor(X: pd.DataFrame, factor: float) -> None:
    """A shifted glucose above the contract's limit would be dead-lettered instead of
    reaching the drift windows, and the alert this switch is meant to trigger never fires."""
    top = X["avg_glucose_level"].max() * factor
    if top > GLUCOSE_LIMIT:
        raise ValueError(
            f"factor {factor} pushes glucose to {top:.0f}, above the contract's {GLUCOSE_LIMIT}"
        )


def events(
    X: pd.DataFrame,
    *,
    seed: int = 42,
    passes: int | None = 1,
    drift_hospital: str | None = None,
    drift_from: int = 0,
    factor: float = 1.0,
    bad_every: int = 0,
    run_id: str = "run",
    pseudonym_key: bytes | None = None,
    clock: Callable[[], float] = time.time,
) -> Iterator[dict]:
    """Every row of X as one event, in a fixed shuffled order, `passes` times (None: forever).

    Glucose is multiplied by `factor` at `drift_hospital` only, starting at event number
    `drift_from`. That number counts every hospital's events, not only the drifting one's:
    it marks a moment in the stream. With `bad_every` = N, every Nth event gets an age the
    contract rejects. Patient ids are keyed with `pseudonym_key` (a fresh random key if None).
    """
    if drift_hospital is not None and drift_hospital not in HOSPITALS:
        raise ValueError(f"unknown hospital {drift_hospital!r}; one of {HOSPITALS}")
    check_factor(X, factor)
    order = X.sample(frac=1, random_state=seed).index
    rows = X.loc[order].to_dict(orient="records")
    hospitals = assign_hospital(X).loc[order].tolist()
    key = pseudonym_key if pseudonym_key is not None else secrets.token_bytes(32)
    patient_ids = [pseudonym(index, key) for index in order]

    n, done = 0, 0
    while passes is None or done < passes:
        for row, hospital, patient_id in zip(rows, hospitals, patient_ids, strict=True):
            if hospital == drift_hospital and n >= drift_from:
                row = {**row, "avg_glucose_level": row["avg_glucose_level"] * factor}
            body = request_body(row, patient_id)
            if bad_every and (n + 1) % bad_every == 0:
                body["age"] = -1  # outside the contract: the worker must dead-letter it
            yield {
                "hospital_id": hospital,
                "event_id": f"{run_id}-{n}",
                "event_time": clock(),
                "patient": body,
            }
            n += 1
        done += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--rate", type=float, default=100, help="target events per second")
    parser.add_argument("--passes", type=int, default=0, help="passes over the data; 0: forever")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-hospital", choices=HOSPITALS)
    parser.add_argument(
        "--drift-from",
        type=int,
        default=0,
        help="event number (all hospitals counted) from which the shift applies",
    )
    parser.add_argument("--factor", type=float, default=1.2, help="glucose multiplier")
    parser.add_argument("--bad-every", type=int, default=0, help="every Nth event invalid")
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    X, _ = load_clean(args.csv)
    factor = args.factor if args.drift_hospital else 1.0
    try:
        check_factor(X, factor)  # before connecting, which can take 30 s if the broker is down
    except ValueError as exc:
        raise SystemExit(f"--factor: {exc}") from None
    stream = events(
        X,
        seed=args.seed,
        passes=args.passes or None,
        drift_hospital=args.drift_hospital,
        drift_from=args.drift_from,
        factor=factor,
        bad_every=args.bad_every,
        run_id=uuid.uuid4().hex[:8],  # fresh event ids on every run
    )
    producer = KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"),
        key_serializer=str.encode,
        value_serializer=lambda v: json.dumps(v, allow_nan=False).encode(),
        acks="all",
        linger_ms=5,
    )
    sent = 0
    refused: list[Exception] = []  # send() is asynchronous: failures arrive in a callback
    interval = 1 / args.rate
    next_at = time.monotonic()
    try:
        for event in stream:
            # The key is the hospital: one hospital's events stay in order on one partition.
            future = producer.send(TOPIC, key=event["hospital_id"], value=event)
            future.add_errback(refused.append)
            sent += 1
            if sent % 1000 == 0:
                log.info("sent %d events", sent)
            # Pace against a schedule, not a fixed sleep, so the time send() takes doesn't
            # slow the stream below the requested rate. After a stall (a slow broker, a
            # suspended laptop), catch up at most one second's worth of events at once.
            next_at = max(next_at + interval, time.monotonic() - 1.0)
            time.sleep(max(0.0, next_at - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()  # every send has now been acknowledged or refused
        producer.close()
        acknowledged = sent - len(refused)
        log.info("done: %d sent, %d acknowledged, %d refused", sent, acknowledged, len(refused))
        if refused:
            log.error("the broker refused %d events; first error: %r", len(refused), refused[0])


if __name__ == "__main__":
    main()
