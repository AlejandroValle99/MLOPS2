"""The model currently being served, swapped as a whole when a new champion loads."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelState:
    model: Any
    name: str
    version: str
    threshold: float
    family: str | None
    data_version: str | None
    loaded_at: float


class ModelHolder:
    """No lock needed: the reloader builds a complete new ModelState and replaces the
    reference in one assignment, so a reader sees either the old state or the new one,
    never a mix. Each RPC reads the state once and uses that copy for the whole call."""

    def __init__(self) -> None:
        self._state: ModelState | None = None

    def get(self) -> ModelState | None:
        return self._state

    def set(self, state: ModelState) -> None:
        self._state = state
