"""services.model_server.reloader: following the champion alias without ever serving a
broken model. The registry lookup and the loader are plain functions here, so no MLflow."""

import threading

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.tracking import MlflowClient

from services.model_server.holder import ModelHolder
from services.model_server.main import WARMUP_ROW
from services.model_server.reloader import Reloader, mlflow_lookup

TAGS = {"threshold": "0.3", "model_family": "logreg", "data_version": "abc123"}


class FakeRegistry:
    """Stands in for MLflow: `champion` is (version, tags), None, or an exception to raise.
    `models` maps a version to what loading it returns (or an exception to raise)."""

    def __init__(self, models: dict) -> None:
        self.champion: tuple[str, dict] | None | Exception = None
        self.models = models
        self.loads: list[str] = []

    def lookup(self) -> tuple[str, dict] | None:
        if isinstance(self.champion, Exception):
            raise self.champion
        return self.champion

    def load(self, version: str):
        self.loads.append(version)
        model = self.models[version]
        if isinstance(model, Exception):
            raise model
        return model


class CannotPredict:
    """Loads fine but fails on its first prediction, like a model whose library is missing."""

    def predict_proba(self, X):
        raise ModuleNotFoundError("No module named 'xgboost'")


@pytest.fixture
def registry(tiny_model):
    return FakeRegistry({"1": tiny_model, "2": tiny_model, "3": OSError("corrupted artifact")})


@pytest.fixture
def holder():
    return ModelHolder()


@pytest.fixture
def reloader(holder, registry):
    return Reloader(holder, "stroke-model", registry.lookup, registry.load, WARMUP_ROW, 0.01)


def test_loads_the_champion_with_its_tags(reloader, registry, holder):
    registry.champion = ("1", TAGS)
    assert reloader.check_once() == "loaded"
    state = holder.get()
    assert state is not None
    assert (state.name, state.version, state.threshold) == ("stroke-model", "1", 0.3)
    assert (state.family, state.data_version) == ("logreg", "abc123")


def test_threshold_defaults_to_one_half_without_a_tag(reloader, registry, holder):
    registry.champion = ("1", {})
    reloader.check_once()
    assert holder.get().threshold == 0.5


def test_the_same_version_is_not_reloaded(reloader, registry):
    registry.champion = ("1", TAGS)
    reloader.check_once()
    assert reloader.check_once() == "unchanged"
    assert registry.loads == ["1"]


def test_follows_the_alias_when_it_moves(reloader, registry, holder):
    registry.champion = ("1", TAGS)
    reloader.check_once()
    registry.champion = ("2", TAGS)
    assert reloader.check_once() == "loaded"
    assert holder.get().version == "2"


def test_a_champion_that_fails_to_load_keeps_the_old_model(reloader, registry, holder):
    registry.champion = ("1", TAGS)
    reloader.check_once()
    registry.champion = ("3", TAGS)  # the corrupted one
    assert reloader.check_once() == "load-failed"
    assert holder.get().version == "1"


def test_a_champion_that_loads_but_cannot_predict_is_refused(reloader, registry, holder):
    registry.models["4"] = CannotPredict()
    registry.champion = ("1", TAGS)
    reloader.check_once()
    registry.champion = ("4", TAGS)
    assert reloader.check_once() == "load-failed"
    assert holder.get().version == "1"


def test_a_bad_threshold_tag_is_refused(reloader, registry, holder):
    registry.champion = ("1", {"threshold": "not-a-number"})
    assert reloader.check_once() == "load-failed"
    assert holder.get() is None


def test_a_known_bad_version_is_not_retried(reloader, registry):
    registry.champion = ("3", TAGS)
    reloader.check_once()
    assert reloader.check_once() == "skipping-known-bad"
    assert registry.loads == ["3"]  # loaded (and logged) once, not every cycle


def test_an_unreachable_registry_keeps_serving(reloader, registry, holder):
    registry.champion = ("1", TAGS)
    reloader.check_once()
    registry.champion = ConnectionError("mlflow is down")
    assert reloader.check_once() == "registry-unreachable"
    assert holder.get().version == "1"


def test_no_champion_yet_loads_nothing(reloader, holder):
    assert reloader.check_once() == "no-champion"
    assert holder.get() is None


def test_on_loaded_runs_once_per_new_model(holder, registry):
    seen = []
    reloader = Reloader(
        holder, "stroke-model", registry.lookup, registry.load, WARMUP_ROW, 0.01, seen.append
    )
    registry.champion = ("1", TAGS)
    reloader.check_once()
    reloader.check_once()  # unchanged: no callback
    registry.champion = ("3", TAGS)
    reloader.check_once()  # failed: no callback
    assert [state.version for state in seen] == ["1"]


def test_run_forever_polls_until_stopped(reloader, registry, holder):
    registry.champion = ("1", TAGS)
    stop = threading.Event()
    thread = threading.Thread(target=reloader.run_forever, args=(stop,))
    thread.start()
    try:
        for _ in range(200):
            if holder.get() is not None:
                break
            stop.wait(0.01)
        assert holder.get().version == "1"
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


# --- the real MLflow lookup: which errors mean "no champion" ------------------------------


@pytest.mark.parametrize("code", [RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE])
def test_mlflow_lookup_reads_a_missing_model_or_alias_as_no_champion(monkeypatch, code):
    def missing(self, name, alias):
        raise MlflowException("not found", error_code=code)

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", missing)
    assert mlflow_lookup("http://unused", "stroke-model", "champion")() is None


def test_mlflow_lookup_raises_when_the_registry_is_broken(monkeypatch):
    def broken(self, name, alias):
        raise MlflowException("connection refused", error_code=INTERNAL_ERROR)

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", broken)
    with pytest.raises(MlflowException):
        mlflow_lookup("http://unused", "stroke-model", "champion")()


def test_mlflow_lookup_returns_the_version_as_a_string(monkeypatch):
    class Version:
        version = 5  # MLflow may hand back an int
        tags = TAGS

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", lambda self, n, a: Version)
    assert mlflow_lookup("http://unused", "stroke-model", "champion")() == ("5", TAGS)
