"""model.train: the candidates, tuning, calibration with an out-of-fold threshold, the logged
run, and the command-line entry point that trains and gates."""

import functools
import json
import uuid
from dataclasses import replace

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import model.train as train
from model.preprocess import build_feature_pipeline
from stroke.config import CANDIDATES, CHAMPION
from stroke.data import frozen_data
from stroke.evaluation import best_threshold
from stroke.registry import log_candidate, trusted_types_for


def xgboost_loads() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:  # on macOS without libomp, xgboost raises its own error, not ImportError
        return False
    return True


@pytest.fixture(scope="module")
def data(raw_csv):
    return frozen_data(raw_csv)


@pytest.fixture
def fast_logging(monkeypatch):
    """Log models without MLflow's pip requirement inference, which takes ~2 s per model."""
    monkeypatch.setattr(
        train, "log_candidate", functools.partial(log_candidate, pip_requirements=[])
    )


@pytest.mark.parametrize("name", CANDIDATES)
def test_every_candidate_takes_raw_rows_and_can_be_served(name, data):
    if name == "xgboost" and not xgboost_loads():
        pytest.skip("xgboost can't load here (on macOS it needs libomp)")
    params = train.fixed_params(name, data.y_train)
    model = train.make_candidate(name, params).fit(data.X_train, data.y_train)
    proba = model.predict_proba(data.X_test.head(20))
    assert proba.shape == (20, 2) and ((proba >= 0) & (proba <= 1)).all()
    trusted_types_for(model)  # only types the serving images can load


def test_age_only_reads_nothing_but_age(data):
    model = train.make_candidate("age_only", {}).fit(data.X_train, data.y_train)
    rows = data.X_test.head(50)
    scrambled = rows.assign(avg_glucose_level=1.0, hypertension=1, smoking_status="smokes")
    assert np.array_equal(model.predict_proba(rows), model.predict_proba(scrambled))


def test_an_unknown_candidate_is_refused():
    with pytest.raises(ValueError, match="unknown candidate"):
        train.make_candidate("knn", {})


def test_xgboost_weighs_positives_by_the_class_ratio(data):
    weight = train.fixed_params("xgboost", data.y_train)["scale_pos_weight"]
    assert weight == pytest.approx((4087 - 199) / 199)


def test_tuning_returns_the_searched_hyperparameters_and_its_score(data):
    params, tuning = train.tune("logreg", data.X_train, data.y_train, n_trials=2)
    assert set(params) == {"C"}
    assert 0 < tuning["cv_pr_auc"] < 1
    assert tuning["tuning_trials"] == 2
    assert train.tune("age_only", data.X_train, data.y_train, n_trials=2) == ({}, {})


def test_calibration_makes_the_probabilities_mean_what_they_say(data):
    """class_weight="balanced" inflates the scores on purpose; after sigmoid calibration the
    average predicted risk is the actual stroke rate again."""
    X, y = data.X_train, data.y_train
    model, oof, threshold = train.calibrate("logreg", {"C": 1.0}, X, y)
    raw = Pipeline(
        [
            ("preprocess", build_feature_pipeline()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ).fit(X, y)
    assert raw.predict_proba(X)[:, 1].mean() > 5 * y.mean()  # about 0.3 against 0.05
    assert oof.mean() == pytest.approx(y.mean(), abs=0.01)
    assert len(oof) == len(X)
    assert threshold == best_threshold(y.to_numpy(), oof)[0]
    assert model.predict_proba(data.X_test)[:, 1].mean() == pytest.approx(y.mean(), abs=0.01)


@pytest.mark.parametrize("name", ["logreg", "random_forest"])
def test_the_logged_model_predicts_on_one_thread_with_the_same_probabilities(name, data):
    params = {"n_estimators": 20, "max_depth": 4} if name == "random_forest" else {}
    model, _, _ = train.calibrate(name, params, data.X_train, data.y_train)
    before = model.predict_proba(data.X_test)
    train.single_threaded(model)
    inner = model.calibrated_classifiers_[0].estimator.named_steps["model"]
    assert inner.n_jobs == 1
    # A forest sums its trees in another order on one thread: only rounding may differ
    assert np.allclose(model.predict_proba(data.X_test), before, rtol=0, atol=1e-12)


def test_a_candidate_run_logs_its_params_metrics_and_artifacts(data, mlflow_store, fast_logging):
    summary = train.train_candidate("logreg", data, n_trials=2, pipeline="test")
    run = MlflowClient(mlflow_store).get_run(summary["run_id"])
    assert set(run.data.params) == {"hp_C"}
    for key in ("f2", "recall", "precision", "roc_auc", "pr_auc", "brier", "ece", "alert_rate"):
        assert {f"oof_{key}", f"test_{key}"} <= set(run.data.metrics)
    assert run.data.metrics["oof_threshold"] == summary["threshold"]
    assert run.data.tags["model_family"] == "logreg"
    assert run.data.tags["test_split"] == data.test_split
    artifacts = {a.path for a in MlflowClient(mlflow_store).list_artifacts(summary["run_id"])}
    assert "reference_stats.json" in artifacts
    loaded = mlflow.sklearn.load_model(summary["model_uri"])
    assert loaded.predict_proba(data.X_test.head(5)).shape == (5, 2)


def test_the_test_split_picks_nothing(data, mlflow_store, fast_logging):
    """Scrambling the test labels may change the test metrics, but never the threshold or
    the out-of-fold score that selects the finalist: those come from training rows only."""
    rng = np.random.default_rng(0)
    scrambled = replace(data, y_test=data.y_test.sample(frac=1, random_state=rng.integers(1e9)))
    a = train.train_candidate("logreg", data, n_trials=2, pipeline="test")
    b = train.train_candidate("logreg", scrambled, n_trials=2, pipeline="test")
    assert a["threshold"] == b["threshold"]
    assert a["oof_f2"] == b["oof_f2"]
    assert a["test_f2"] != b["test_f2"]


def test_the_finalist_is_the_best_out_of_fold_f2_not_the_best_test_f2():
    summaries = [
        {"model_family": "a", "oof_f2": 0.40, "test_f2": 0.45},
        {"model_family": "b", "oof_f2": 0.42, "test_f2": 0.39},
    ]
    assert train.select_finalist(summaries)["model_family"] == "b"


def test_the_cli_trains_and_gates_then_skips_once_a_champion_exists(
    raw_csv, mlflow_store, fast_logging, monkeypatch, capsys
):
    name = f"stroke-cli-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(train, "MODEL_NAME", name)
    args = ["--data", str(raw_csv), "--trials", "1", "--candidates", "age_only,logreg"]

    train.main([*args, "--skip-if-champion"])  # nothing serves yet: trains anyway
    result = json.loads(capsys.readouterr().out)
    assert (result["decision"], result["reason"]) == ("promoted", "no champion yet")
    client = MlflowClient(mlflow_store)
    champion = client.get_model_version_by_alias(name, CHAMPION)
    assert champion.tags["model_family"] == result["model_family"]
    assert 0 < float(champion.tags["threshold"]) < 1
    assert len(champion.tags["artifact_sha256"]) == 64
    card = mlflow.artifacts.load_dict(f"runs:/{champion.run_id}/model_card.json")
    assert card["model"] == f"{name} v{champion.version}"
    assert card["gate"]["decision"] == "promoted"

    train.main([*args, "--skip-if-champion"])  # compose's model-init on a later start
    assert capsys.readouterr().out == ""
