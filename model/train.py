"""Train the stroke prediction Random Forest and log the run to MLflow."""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model.preprocess import build_feature_pipeline, load_data
from stroke.data import data_version_for

DATA_PATH = os.getenv("DATA_PATH", "data/healthcare-dataset-stroke-data.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "stroke-prediction"
REGISTERED_MODEL_NAME = "stroke-model"
CHAMPION_ALIAS = "champion"

BETA = 2

RF_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 19,
    "max_features": "log2",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

VAL_RATIO = 0.20
TEST_RATIO = 0.20


def _get_or_create_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return str(
            mlflow.create_experiment(
                name=name,
                tags={"project": "stroke-prediction", "team": "mlops-fiuba"},
            )
        )
    return str(experiment.experiment_id)


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        f"f{BETA}": fbeta_score(y, y_pred, beta=BETA),
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
    }


def train() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = _get_or_create_experiment(EXPERIMENT_NAME)

    X, y = load_data(DATA_PATH)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=y,
        random_state=42,
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("preprocess", build_feature_pipeline()),
            ("model", RandomForestClassifier(**RF_PARAMS)),
        ]
    )
    pipeline.fit(X_train, y_train)

    val_metrics = _evaluate(pipeline, X_val, y_val)
    test_metrics = _evaluate(pipeline, X_test, y_test)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="random_forest_training",
        tags={"model": "random_forest"},
    ):
        mlflow.log_params(RF_PARAMS)
        mlflow.log_params(
            {
                "val_ratio": VAL_RATIO,
                "test_ratio": TEST_RATIO,
                "train_samples": len(X_train),
                "beta": BETA,
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.sklearn.log_model(
            pipeline,
            name="stroke-model",
            input_example=X_train.head(1),
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "model.preprocess.BMIGroupImputer",
                "model.preprocess.StrokeCleaner",
                "sklearn.tree._tree.Tree",
            ],
        )

        # Promote the latest version to the champion alias
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'",
            max_results=1,
            order_by=["version_number DESC"],
        )
        if versions:
            new_version = versions[0].version
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version)

            # D5 (docs/PLAN.md): tag model_family and data_version on every version, so a
            # version can always be traced to the algorithm and the exact data that made
            # it. `threshold` is NOT tagged here on purpose — it depends on Phase 5's
            # calibration + out-of-fold threshold selection, which doesn't exist yet.
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME, new_version, "model_family", "random_forest"
            )
            version = data_version_for(DATA_PATH)
            if version is not None:
                client.set_model_version_tag(
                    REGISTERED_MODEL_NAME, new_version, "data_version", version
                )

            print(
                f"Modelo '{REGISTERED_MODEL_NAME}' v{new_version} "
                f"promovido a alias '{CHAMPION_ALIAS}'"
            )

    print("=== Validación ===")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=== Test ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nExperimento '{EXPERIMENT_NAME}' → {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    train()
