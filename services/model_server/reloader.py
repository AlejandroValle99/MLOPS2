"""Hot reload: follow the registry's champion alias and swap the served model safely.

The registry lookup and the model loader are injected, so tests drive the reloader with
plain functions and never need MLflow. mlflow_lookup and mlflow_loader below are the real
ones the server uses.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from services.model_server.holder import ModelHolder, ModelState
from stroke.registry import load_version

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5

# What MLflow answers when there is simply no champion yet: the registered model does not
# exist, or it exists without the alias. Any other error means the registry itself is
# unreachable or broken, which must not be mistaken for "nothing to serve".
NO_CHAMPION_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}

Lookup = Callable[[], tuple[str, dict] | None]  # -> (version, tags), or None: no champion
Loader = Callable[[str], Any]  # version -> fitted model


class Reloader:
    def __init__(
        self,
        holder: ModelHolder,
        name: str,
        lookup: Lookup,
        loader: Loader,
        warmup_row: dict,
        interval_s: float,
        on_loaded: Callable[[ModelState], None] | None = None,
    ) -> None:
        self.holder, self.name = holder, name
        self.lookup, self.loader = lookup, loader
        self.warmup_row, self.interval_s = warmup_row, interval_s
        self.on_loaded = on_loaded
        self.bad_versions: set[str] = set()

    def check_once(self) -> str:
        """One poll of the registry. Returns a short status, which is what tests assert on."""
        try:
            found = self.lookup()
        except Exception:
            return "registry-unreachable"  # keep serving whatever we have
        if found is None:
            return "no-champion"
        version, tags = found

        current = self.holder.get()
        if current is not None and current.version == version:
            return "unchanged"
        if version in self.bad_versions:
            return "skipping-known-bad"  # already logged once; don't retry every cycle

        try:
            model = self.loader(version)
            # Warm-up: one real prediction before the swap. A model that loads but cannot
            # score (a missing library, a broken artifact, a wrong input schema) fails here,
            # while the old model is still the one serving.
            model.predict_proba(pd.DataFrame([self.warmup_row]))
            state = ModelState(
                model=model,
                name=self.name,
                version=version,
                threshold=float(tags.get("threshold", DEFAULT_THRESHOLD)),
                family=tags.get("model_family"),
                data_version=tags.get("data_version"),
                loaded_at=time.time(),
            )
        except Exception:
            log.exception("version %s failed to load or warm up; keeping the old model", version)
            self.bad_versions.add(version)
            return "load-failed"

        self.holder.set(state)  # the swap: one assignment
        log.info("now serving %s version %s", self.name, version)
        if self.on_loaded is not None:
            self.on_loaded(state)
        return "loaded"

    def run_forever(self, stop: threading.Event) -> None:
        last = None
        while not stop.is_set():
            status = self.check_once()
            if status != last:  # log changes only, not "unchanged" every interval
                log.info("reload check: %s", status)
                last = status
            stop.wait(self.interval_s)


def mlflow_lookup(tracking_uri: str, name: str, alias: str) -> Lookup:
    client = MlflowClient(tracking_uri)

    def lookup() -> tuple[str, dict] | None:
        try:
            mv = client.get_model_version_by_alias(name, alias)
        except MlflowException as exc:
            if exc.error_code in NO_CHAMPION_CODES:
                return None
            raise
        return str(mv.version), dict(mv.tags)  # MLflow may return the version as an int

    return lookup


def mlflow_loader(tracking_uri: str, name: str) -> Loader:
    mlflow.set_tracking_uri(tracking_uri)

    def load(version: str) -> Any:
        # An explicit version, never models:/name@champion: the alias could move between
        # the lookup above and this load, and then the state would describe the wrong model.
        # load_version checks the files against the version's artifact_sha256 tag first, so
        # a tampered artifact raises here and the reloader keeps the old model ("load-failed").
        return load_version(name, version)

    return load
