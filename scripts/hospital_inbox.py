"""One hospital's side of the stream: read its own scores and list the high-risk patients.

    uv run --group stream python scripts/hospital_inbox.py rural-60plus

This stands in for the hospital's system receiving its results. All hospitals share one
`predictions` topic here, keyed by hospital, so each inbox drops the others' messages; a
real deployment would give each hospital its own topic, readable only with its own
credentials.
"""

import json
import os
import sys
from collections.abc import Iterable
from typing import Any

from kafka import KafkaConsumer


def follow_ups(hospital_id: str, records: Iterable[Any]) -> list[dict]:
    """The high-risk scores addressed to this hospital. Pure, so it's testable without a broker."""
    key = hospital_id.encode()
    return [r.value for r in records if r.key == key and r.value["high_risk"]]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: hospital_inbox.py <hospital_id>   (e.g. rural-60plus)")
    hospital_id = sys.argv[1]
    consumer = KafkaConsumer(
        "predictions",
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"),
        group_id=f"hospital-{hospital_id}",  # its own group: reads every score, whatever others do
        # The first run starts from now, not from the topic's start. Later runs pick up where
        # the last one stopped (the group remembers its position), so scores published while
        # the inbox was closed are still delivered.
        auto_offset_reset="latest",
        value_deserializer=json.loads,  # keys stay bytes: a keyless message can't crash this
    )
    print(f"{hospital_id}: waiting for high-risk patients (Ctrl-C to stop)")
    try:
        for record in consumer:
            for p in follow_ups(hospital_id, [record]):
                print(
                    f"follow up {p['patient_id']}: p={p['probability']:.2f} "
                    f"(threshold {p['threshold']:.2f}, model v{p['model_version']})"
                )
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
