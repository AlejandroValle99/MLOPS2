"""stroke.registry: the gate, promotion and rollback against a real (sqlite) MLflow, the
artifact digest checked before every load, the audit lines, and what the model server's
reloader sees after each of them."""

import json
import logging
import uuid
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skops.io import dumps

from model.preprocess import build_feature_pipeline
from services.model_server.holder import ModelHolder
from services.model_server.main import WARMUP_ROW
from services.model_server.reloader import Reloader, mlflow_loader, mlflow_lookup
from stroke.config import CHAMPION, PREVIOUS_CHAMPION
from stroke.registry import (
    DIGEST_TAG,
    ArtifactIntegrityError,
    digest_of,
    get_alias,
    load_version,
    log_candidate,
    promote,
    record_missing_digests,
    register,
    rollback,
    run_gate,
    trusted_types_for,
    use_experiment,
)

SPLIT = "abc123-seed42-test0.2"


@pytest.fixture
def name(mlflow_store) -> str:
    """A registered-model name of its own for each test."""
    return f"stroke-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def client(mlflow_store) -> MlflowClient:
    return MlflowClient(mlflow_store)


@pytest.fixture
def new_version(name, tiny_model, split):
    """Logs `model` (tiny_model by default) and registers it with the given tags."""
    X_train = split[0]

    def make(threshold: float, test_split: str = SPLIT, model=None) -> str:
        use_experiment()
        logged = log_candidate(
            "logreg",
            tiny_model if model is None else model,
            params={},
            metrics={},
            tags={},
            reference_stats={},
            input_example=X_train.head(1),
            pip_requirements=[],  # skip MLflow's requirement inference: ~2 s per model
        )
        finalist = {
            "model_uri": logged.model_uri,
            "artifact_sha256": logged.artifact_sha256,
            "threshold": threshold,
            "model_family": "logreg",
            "data_version": "abc123",
            "test_split": test_split,
        }
        return register(name, finalist)

    return make


@pytest.fixture
def gate(client, name, split):
    _, X_test, _, y_test = split

    def run(version: str, **kwargs) -> dict:
        return run_gate(client, name, version, X_test, y_test, n_bootstraps=500, **kwargs)

    return run


def aliases(client, name) -> dict[str, str]:
    return {a: str(v) for a, v in client.get_registered_model(name).aliases.items()}


def test_register_sets_the_serving_tags_before_any_alias(client, name, new_version):
    version = new_version(threshold=0.42)
    tags = client.get_model_version(name, version).tags
    assert tags["threshold"] == "0.42"
    assert tags["model_family"] == "logreg"
    assert tags["test_split"] == SPLIT
    assert aliases(client, name) == {}  # registering promotes nothing


def test_the_first_model_is_promoted(client, name, new_version, gate):
    version = new_version(threshold=0.5)
    result = gate(version)
    assert (result["decision"], result["reason"]) == ("promoted", "no champion yet")
    assert aliases(client, name) == {CHAMPION: version}  # no previous champion to keep
    assert client.get_model_version(name, version).tags["gate_decision"] == "promoted"


def test_a_weaker_challenger_is_rejected_and_the_champion_stays(client, name, new_version, gate):
    champion = new_version(threshold=0.5)
    gate(champion)
    weaker = new_version(threshold=0.99)  # the same model, but it flags almost nobody
    result = gate(weaker)
    assert result["decision"] == "rejected"
    assert result["p_better"] < 0.05
    assert result["challenger_f2"] < result["champion_f2"]
    assert aliases(client, name) == {CHAMPION: champion}
    tags = client.get_model_version(name, weaker).tags
    assert tags["gate_decision"] == "rejected"
    assert tags["gate_champion"] == champion
    assert float(tags["gate_p_better"]) < 0.05
    assert {"gate_reason", "gate_at", "gate_mean_diff", "gate_ci90"} <= set(tags)


def test_an_identical_challenger_is_rejected_because_a_tie_is_not_a_win(
    client, name, new_version, gate
):
    gate(new_version(threshold=0.5))
    result = gate(new_version(threshold=0.5))
    assert (result["decision"], result["p_better"]) == ("rejected", 0.0)


def test_a_forced_promotion_keeps_the_old_champion_as_previous(client, name, new_version, gate):
    old = new_version(threshold=0.5)
    gate(old)
    new = new_version(threshold=0.99)
    result = gate(new, min_p_better=0.0)  # every P(better) passes: the documented override
    assert result["decision"] == "promoted"
    assert aliases(client, name) == {CHAMPION: new, PREVIOUS_CHAMPION: old}
    assert client.get_model_version(name, new).tags["gate_min_p_better"] == "0.0"


def test_rollback_restores_the_previous_champion_with_its_threshold(
    client, name, new_version, gate, mlflow_store
):
    old = new_version(threshold=0.3)
    gate(old)
    new = new_version(threshold=0.7)
    gate(new, min_p_better=0.0)

    # The model server's own reloader, pointed at this registry
    holder = ModelHolder()
    reloader = Reloader(
        holder,
        name,
        lookup=mlflow_lookup(mlflow_store, name, CHAMPION),
        loader=mlflow_loader(mlflow_store, name),
        warmup_row=WARMUP_ROW,
        interval_s=1,
    )
    assert reloader.check_once() == "loaded"
    state = holder.get()
    assert state is not None and (state.version, state.threshold) == (new, 0.7)

    assert rollback(client, name) == old
    assert aliases(client, name) == {CHAMPION: old}  # previous_champion is used up
    assert "rolled_back_at" in client.get_model_version(name, new).tags
    assert reloader.check_once() == "loaded"
    state = holder.get()
    assert state is not None and (state.version, state.threshold) == (old, 0.3)

    with pytest.raises(LookupError):  # a second rollback must not flip back to `new`
        rollback(client, name)


def test_a_rollback_says_who_asked_on_the_version_and_in_the_audit_log(
    client, name, new_version, gate, caplog
):
    old = new_version(threshold=0.3)
    gate(old)
    new = new_version(threshold=0.7)
    gate(new, min_p_better=0.0)
    caplog.set_level(logging.INFO, logger="audit")
    rollback(client, name, actor="ops-1")
    assert client.get_model_version(name, new).tags["rolled_back_by"] == "ops-1"
    (line,) = audit_lines(caplog)
    assert (line["actor"], line["action"]) == ("ops-1", "rollback")
    assert (line["from_version"], line["to_version"]) == (new, old)


def test_rollback_without_a_previous_champion_fails(client, name, new_version, gate):
    gate(new_version(threshold=0.5))
    with pytest.raises(LookupError):
        rollback(client, name)


def test_promoting_the_champion_again_does_not_make_it_its_own_previous(client, name, new_version):
    version = new_version(threshold=0.5)
    promote(client, name, version)
    promote(client, name, version)
    assert aliases(client, name) == {CHAMPION: version}


def test_a_champion_from_before_the_gate_is_replaced_without_a_comparison(
    client, name, new_version, gate
):
    use_experiment()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            Pipeline([("model", LogisticRegression())]), name="model", pip_requirements=[]
        )
    legacy = str(mlflow.register_model(info.model_uri, name, tags={"model_family": "x"}).version)
    promote(client, name, legacy)  # how the old trainer put it there: no gate, no test_split

    challenger = new_version(threshold=0.5)
    result = gate(challenger)
    assert result["decision"] == "promoted"
    assert "nothing to compare against" in result["reason"]
    assert "p_better" not in result  # nothing was measured, and nothing claims it was
    assert aliases(client, name) == {CHAMPION: challenger, PREVIOUS_CHAMPION: legacy}


def test_a_champion_judged_on_another_test_split_is_not_compared(client, name, new_version, gate):
    gate(new_version(threshold=0.5, test_split="other-data-seed42-test0.2"))
    result = gate(new_version(threshold=0.99))  # would lose a comparison
    assert result["decision"] == "promoted"
    assert "other-data-seed42-test0.2" in result["reason"]


def test_the_gate_refuses_a_version_without_its_serving_tags(client, name, gate):
    use_experiment()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            Pipeline([("model", LogisticRegression())]), name="model", pip_requirements=[]
        )
    version = str(mlflow.register_model(info.model_uri, name).version)
    with pytest.raises(ValueError, match="threshold"):
        gate(version)
    assert get_alias(client, name) is None


def test_trusted_types_are_the_calibrated_pipelines_own(split):
    X_train, _, y_train, _ = split
    model = CalibratedClassifierCV(
        Pipeline([("preprocess", build_feature_pipeline()), ("model", LogisticRegression())]),
        method="sigmoid",
        cv=3,
        ensemble=False,
    ).fit(X_train, y_train)
    assert set(trusted_types_for(model)) == {
        "model.preprocess.BMIGroupImputer",
        "model.preprocess.StrokeCleaner",
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
    }


class Doubler(BaseEstimator, TransformerMixin):
    """A transformer the serving images don't have (it lives in this test module)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * 2


def test_a_model_with_a_type_serving_cannot_load_is_refused_before_it_is_logged(split):
    X_train, _, y_train, _ = split
    model = Pipeline([("double", Doubler()), ("model", LogisticRegression())])
    model.fit(X_train[["age"]], y_train)
    with pytest.raises(ValueError, match="Doubler"):
        trusted_types_for(model)


# --- the audit log ---------------------------------------------------------------------------


def audit_lines(caplog) -> list[dict]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "audit"]


def test_every_gate_decision_leaves_one_audit_line(client, name, new_version, gate, caplog):
    caplog.set_level(logging.INFO, logger="audit")
    champion = new_version(threshold=0.5)
    gate(champion)
    weaker = new_version(threshold=0.99)
    gate(weaker)
    promoted, rejected = audit_lines(caplog)
    assert (promoted["actor"], promoted["action"], promoted["version"]) == (
        "gate",
        "promote",
        champion,
    )
    assert (rejected["action"], rejected["version"], rejected["champion"]) == (
        "reject",
        weaker,
        champion,
    )
    assert rejected["p_better"] < 0.05 and rejected["reason"]


# --- the artifact digest ---------------------------------------------------------------------


def artifact_dir(client: MlflowClient, name: str, version: str) -> Path:
    """Where the version's files really are: here a folder, in compose a MinIO prefix."""
    model_id = client.get_model_version(name, version).source.rsplit("/", 1)[-1]
    return Path(urlparse(client.get_logged_model(model_id).artifact_location).path)


def test_the_digest_taken_at_logging_is_the_registered_versions(client, name, new_version):
    version = new_version(threshold=0.5)
    recorded = client.get_model_version(name, version).tags[DIGEST_TAG]
    assert len(recorded) == 64
    # A download of the version holds one more file than was logged (registered_model_meta,
    # which names the version); the digest leaves it out, so the two still agree.
    assert digest_of(f"models:/{name}/{version}") == recorded


def test_a_verified_version_loads_and_scores(name, new_version, tiny_model):
    version = new_version(threshold=0.5)
    frame = pd.DataFrame([WARMUP_ROW])
    loaded = load_version(name, version)
    assert np.array_equal(loaded.predict_proba(frame), tiny_model.predict_proba(frame))


def test_a_swapped_model_file_is_refused_and_the_old_model_keeps_serving(
    client, name, new_version, gate, mlflow_store, tiny_model, split
):
    X_train, _, y_train, _ = split
    first = new_version(threshold=0.5)
    gate(first)
    holder = ModelHolder()
    reloader = Reloader(
        holder,
        name,
        lookup=mlflow_lookup(mlflow_store, name, CHAMPION),
        loader=mlflow_loader(mlflow_store, name),
        warmup_row=WARMUP_ROW,
        interval_s=1,
    )
    assert reloader.check_once() == "loaded"

    second = new_version(threshold=0.5)
    gate(second, min_p_better=0.0)
    # The attack: a different, working model in place of the registered one. Same classes,
    # so it would load and score without complaint; here it is trained on inverted labels.
    impostor = clone(tiny_model).fit(X_train, 1 - y_train)
    (artifact_dir(client, name, second) / "model.skops").write_bytes(dumps(impostor))
    unchecked = mlflow.sklearn.load_model(f"models:/{name}/{second}")
    assert unchecked.predict_proba(pd.DataFrame([WARMUP_ROW]))[0, 1] < 0.5  # it would serve

    with pytest.raises(ArtifactIntegrityError, match="changed since registration"):
        load_version(name, second)
    assert reloader.check_once() == "load-failed"
    state = holder.get()
    assert state is not None and state.version == first  # still the verified model


def test_an_edited_mlmodel_is_refused_too(client, name, new_version):
    """MLmodel lists the types skops will trust: editing it is tampering as well."""
    version = new_version(threshold=0.5)
    mlmodel = artifact_dir(client, name, version) / "MLmodel"
    mlmodel.write_text(mlmodel.read_text() + "\n# edited\n")
    with pytest.raises(ArtifactIntegrityError):
        load_version(name, version)


def legacy_version(name: str) -> str:
    """A version registered the way the old trainer did: no digest, no serving tags."""
    use_experiment()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            Pipeline([("model", LogisticRegression())]), name="model", pip_requirements=[]
        )
    return str(mlflow.register_model(info.model_uri, name).version)


def test_a_version_without_a_digest_is_refused(name):
    version = legacy_version(name)
    with pytest.raises(ArtifactIntegrityError, match="no artifact_sha256"):
        load_version(name, version)


def test_the_backfill_records_a_digest_only_where_one_is_missing(client, name, new_version, caplog):
    current = new_version(threshold=0.5)
    before = client.get_model_version(name, current).tags[DIGEST_TAG]
    old = legacy_version(name)
    caplog.set_level(logging.INFO, logger="audit")

    assert record_missing_digests(client, name, actor="ops-1") == [old]
    assert client.get_model_version(name, old).tags[DIGEST_TAG] == digest_of(
        f"models:/{name}/{old}"
    )
    assert client.get_model_version(name, current).tags[DIGEST_TAG] == before  # untouched
    (line,) = audit_lines(caplog)
    assert (line["actor"], line["action"], line["version"]) == ("ops-1", "record-digest", old)
    assert record_missing_digests(client, name, actor="ops-1") == []  # nothing left to do
