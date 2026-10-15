"""The stream's input contract, and the check every event passes before it is scored.

An intake event is what a hospital's system publishes when a patient is registered: who
sent it, when, and the patient, as exactly the body the REST API accepts. Reusing the
REST contract means both channels accept and reject the same patients.
"""

from collections.abc import Collection, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from api.schemas import StrokeInput


class IntakeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hospital_id: str = Field(min_length=1, max_length=64)
    event_id: str = Field(min_length=1, max_length=128)
    event_time: float = Field(gt=0, allow_inf_nan=False)  # unix seconds, set by the sender
    patient: StrokeInput


def _describe(exc: ValidationError) -> str:
    """One line per problem, e.g. "patient.age: Input should be greater than 0"."""
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'event'}: {err['msg']}"
        for err in exc.errors(include_url=False)
    )


def parse(raw: bytes | None, known_hospitals: Collection[str]) -> IntakeEvent:
    """Raw message bytes -> a valid event. ValueError, with a readable reason, otherwise:
    for bytes that aren't JSON, a schema failure, or a hospital the service doesn't know."""
    if raw is None:  # a message with no value at all
        raise ValueError("event: empty message")
    try:
        event = IntakeEvent.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(_describe(exc)) from None
    if event.hospital_id not in known_hospitals:
        raise ValueError(f"hospital_id: unknown hospital {event.hospital_id!r}")
    return event


def split_events(
    records: Sequence[Any], known_hospitals: Collection[str]
) -> tuple[list[tuple[Any, IntakeEvent]], list[tuple[Any, str]]]:
    """Kafka records -> (record, event) pairs that passed, and (record, reason) pairs that
    did not. The record travels along so a rejected event can be dead-lettered as received.

    A message keyed by one hospital must carry that hospital's event: the key decides the
    partition, so a mismatch would interleave one hospital's patients with another's."""
    valid, rejected = [], []
    for record in records:
        try:
            event = parse(record.value, known_hospitals)
            if record.key is not None and record.key != event.hospital_id.encode():
                raise ValueError(
                    f"hospital_id: {event.hospital_id!r} does not match the message key "
                    f"{record.key.decode('utf-8', errors='replace')!r}"
                )
            valid.append((record, event))
        except ValueError as exc:
            rejected.append((record, str(exc)))
    return valid, rejected
