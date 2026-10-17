"""Train the stroke-risk candidates, pick a finalist, and send it through the promotion gate.

    python -m model.train [--trials N] [--candidates logreg,xgboost] [--skip-if-champion]

For each candidate family, on the training split only:

1. Tune: Optuna (TPE sampler, median pruner) maximizes the mean 5-fold CV PR-AUC. PR-AUC
   doesn't depend on a threshold, so it is a steadier target than F2 with ~200 positives.
2. Calibrate: sigmoid calibration on out-of-fold predictions. Class weights push the
   scores up on purpose; after calibration a 0.2 means about a 20% chance.
3. Choose the threshold: the cut that maximizes F2 on out-of-fold calibrated probabilities,
   so every training row is scored by a model that never saw it.
4. Log: the run, its hyperparameters, out-of-fold and test metrics, and the model, whose
   files are hashed right after the upload (the digest the model server checks later).

The candidate with the best out-of-fold F2 is the finalist. It becomes a new version of the
served model and goes through the gate (stroke.registry.run_gate), which promotes it only
with statistical evidence that it beats the champion, and every version gets a model card.
The test split picks nothing: not the hyperparameters, not the threshold, not the finalist.
It is scored once per candidate for the record, and the gate reads it.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.preprocess import build_feature_pipeline
from stroke.config import (
    CANDIDATES,
    CV_FOLDS,
    GATE_MIN_P_BETTER,
    MODEL_NAME,
    RANDOM_SEED,
)
from stroke.data import TrainingData, assign_hospital, frozen_data
from stroke.drift import reference_document
from stroke.evaluation import PRIMARY, best_threshold, metrics_at
from stroke.model_card import log_model_card
from stroke.registry import get_alias, log_candidate, register, run_gate, use_experiment

log = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH", "data/raw/healthcare-dataset-stroke-data.csv")


def search_space(name: str, trial: optuna.Trial) -> dict:
    """The hyperparameters Optuna tunes for each family. age_only has none."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "random_forest":
        # Shallow-ish trees with large leaves: with ~200 positives, deep trees memorize them,
        # and they also make the artifact large and every prediction slow.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    return {}


def fixed_params(name: str, y: pd.Series) -> dict:
    """Hyperparameters set from the data rather than tuned (logged alongside the tuned ones)."""
    if name == "xgboost":
        # XGBoost's form of class weights: each positive counts as much as neg/pos negatives.
        return {"scale_pos_weight": float((y == 0).sum() / (y == 1).sum())}
    return {}


def make_candidate(name: str, params: dict) -> Pipeline:
    """An unfitted pipeline, raw DataFrame in, probability out. Built only from scikit-learn,
    xgboost and model.preprocess classes, because those are what the serving images have."""
    if name == "age_only":
        # The floor: a logistic regression on age alone.
        return Pipeline(
            [
                ("preprocess", ColumnTransformer([("age", StandardScaler(), ["age"])])),
                ("model", LogisticRegression(**params)),
            ]
        )
    model: Any
    if name == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", **params)
    elif name == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, **params
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier  # imported here: on macOS it needs libomp to load

        model = XGBClassifier(eval_metric="aucpr", random_state=RANDOM_SEED, n_jobs=-1, **params)
    else:
        raise ValueError(f"unknown candidate {name!r}; expected one of {CANDIDATES}")
    return Pipeline([("preprocess", build_feature_pipeline()), ("model", model)])


def tune(
    name: str, X: pd.DataFrame, y: pd.Series, n_trials: int, seed: int = RANDOM_SEED
) -> tuple[dict, dict[str, float]]:
    """(best hyperparameters, tuning metrics) by mean CV PR-AUC on the training rows."""
    fixed = fixed_params(name, y)
    if name == "age_only":  # nothing to tune
        return fixed, {}
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = {**search_space(name, trial), **fixed}
        scores: list[float] = []
        for step, (fit_rows, val_rows) in enumerate(folds):
            model = make_candidate(name, params).fit(X.iloc[fit_rows], y.iloc[fit_rows])
            proba = model.predict_proba(X.iloc[val_rows])[:, 1]
            scores.append(average_precision_score(y.iloc[val_rows], proba))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():  # well below earlier trials after the same folds
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials)
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    return {**study.best_params, **fixed}, {
        "cv_pr_auc": study.best_value,
        "tuning_trials": float(len(study.trials)),
        "tuning_pruned": float(pruned),
    }


def calibrate(
    name: str, params: dict, X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_SEED
) -> tuple[CalibratedClassifierCV, np.ndarray, float]:
    """(calibrated model fitted on all rows, out-of-fold calibrated probabilities, threshold).

    Sigmoid, not isotonic: isotonic overfits with this few positives. ensemble=False keeps
    one fitted pipeline plus its calibrator instead of five copies: a fifth of the size and
    of the time per prediction, with no measurable loss here.

    The out-of-fold probabilities come from 5 x 5 fits: each outer fold calibrates on its
    own training rows only, then scores rows it never saw. The threshold is chosen on those.
    """
    calibrated = CalibratedClassifierCV(
        make_candidate(name, params), method="sigmoid", cv=CV_FOLDS, ensemble=False
    )
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(calibrated, X, y, cv=folds, method="predict_proba")[:, 1]
    threshold, _ = best_threshold(y.to_numpy(), oof)
    return calibrated.fit(X, y), oof, threshold


def single_threaded(model: CalibratedClassifierCV) -> CalibratedClassifierCV:
    """The fitted model set to predict on one thread. Training uses every core, but the
    model server scores one patient (or a few hundred) per call, where starting a thread
    pool costs more than it saves: a 100-tree forest took 16.7 ms per row with n_jobs=-1
    and 4.8 ms with n_jobs=1. The probabilities stay the same, up to float rounding."""
    for calibrated in model.calibrated_classifiers_:
        estimator = calibrated.estimator.named_steps["model"]
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)
    return model


def train_candidate(
    name: str,
    data: TrainingData,
    n_trials: int,
    pipeline: str,
) -> dict:
    """Tune, calibrate, threshold, score and log one candidate. Returns a small summary
    (it travels through XCom), including what register() needs if it is the finalist."""
    log.info("training %s on %d rows", name, len(data.X_train))
    params, tuning = tune(name, data.X_train, data.y_train, n_trials)
    model, oof, threshold = calibrate(name, params, data.X_train, data.y_train)
    oof_metrics = metrics_at(data.y_train, oof, threshold)
    # The only time this candidate sees the test set: scored once, for the record.
    test_metrics = metrics_at(data.y_test, model.predict_proba(data.X_test)[:, 1], threshold)

    use_experiment()
    logged = log_candidate(
        name,
        single_threaded(model),
        params,
        metrics={
            **{f"oof_{k}": v for k, v in oof_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **tuning,
        },
        tags={
            "pipeline": pipeline,
            "data_version": data.data_version,
            "test_split": data.test_split,
        },
        reference_stats=reference_document(
            data.X_train, assign_hospital(data.X_train), data.data_version
        ),
        input_example=data.X_train.head(1),
    )
    summary = {
        "model_family": name,
        "run_id": logged.run_id,
        "model_uri": logged.model_uri,
        "artifact_sha256": logged.artifact_sha256,
        "threshold": threshold,
        "data_version": data.data_version,
        "test_split": data.test_split,
        f"oof_{PRIMARY}": oof_metrics[PRIMARY],
        f"test_{PRIMARY}": test_metrics[PRIMARY],
    }
    log.info("%s: %s", name, json.dumps(summary))
    return summary


def select_finalist(summaries: list[dict]) -> dict:
    """The candidate with the best out-of-fold F2: training-only evidence picks it."""
    return max(summaries, key=lambda s: s[f"oof_{PRIMARY}"])


def register_and_gate(
    summaries: list[dict],
    data: TrainingData,
    min_p_better: float = GATE_MIN_P_BETTER,
    name: str = MODEL_NAME,
) -> dict:
    """Register the finalist as a new version, let the gate decide on it, then write the
    version's model card (it records the gate's decision, so it comes last)."""
    finalist = select_finalist(summaries)
    version = register(name, finalist)
    log.info("finalist %s registered as %s v%s", finalist["model_family"], name, version)
    client = MlflowClient()
    result = run_gate(client, name, version, data.X_test, data.y_test, min_p_better)
    try:
        log_model_card(client, name, version)
    except Exception:
        # The gate has already decided, and promoted if it passed: failing the run here
        # would read as a failed promotion. The decision is on the version's tags anyway.
        log.exception("%s v%s: could not write its model card", name, version)
    return {**result, "model_family": finalist["model_family"]}


def train_and_gate(
    data: TrainingData,
    n_trials: int,
    pipeline: str,
    candidates: tuple[str, ...] = CANDIDATES,
    min_p_better: float = GATE_MIN_P_BETTER,
    name: str = MODEL_NAME,
) -> dict:
    """Every candidate, one after the other, then the gate. The comparison DAG does the
    same in parallel tasks; this is for model-init and the retrain DAG."""
    summaries = [train_candidate(family, data, n_trials, pipeline) for family in candidates]
    return register_and_gate(summaries, data, min_p_better, name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the candidates, pick a finalist, and send it through the gate."
    )
    parser.add_argument("--data", default=DATA_PATH, help="the raw CSV (with its manifest)")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per candidate")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--min-p-better", type=float, default=GATE_MIN_P_BETTER)
    parser.add_argument(
        "--skip-if-champion",
        action="store_true",
        help="exit without training when a champion already exists (the compose bootstrap)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # MLflow reads MLFLOW_TRACKING_URI from the environment.
    name = MODEL_NAME
    if args.skip_if_champion and get_alias(MlflowClient(), name) is not None:
        log.info("%s already has a champion; nothing to bootstrap", name)
        return
    result = train_and_gate(
        frozen_data(Path(args.data)),
        n_trials=args.trials,
        pipeline="train.py",
        candidates=tuple(args.candidates.split(",")),
        min_p_better=args.min_p_better,
        name=name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
