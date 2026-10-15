"""DAG etl_train_models_process_taskflow: train every candidate, pick a finalist, gate it.

    verify_raw ─┬─ train_age_only ──────┐
                ├─ train_logreg ────────┤
                ├─ train_random_forest ─┼─ register_and_gate
                └─ train_xgboost ───────┘

verify_raw checks the raw file against its manifest. Each train_* task rebuilds the frozen
split from that same file, then tunes, calibrates and thresholds one candidate and logs it to
MLflow (model/train.py has the details). register_and_gate registers the candidate with the
best out-of-fold F2 as a new version of the served model, and the gate promotes it only if
it beats the champion on the test set in at least `gate_min_p_better` of the bootstrap
resamples. The decision and its numbers are tagged on the version either way.

Trigger it by hand ("Trigger DAG w/ config" changes the parameters). Setting
gate_min_p_better to 0 promotes the finalist whatever the comparison says; that is still
a gate decision, recorded on the version like any other.

Only small values cross XCom (the data version, run ids, metrics), never data or models.
Heavy imports live inside the tasks, so the scheduler parses this file quickly.
"""

import datetime
import os
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

from stroke.config import CANDIDATES, GATE_MIN_P_BETTER

DATA_PATH = os.getenv("DATA_PATH", "/opt/airflow/data/raw/healthcare-dataset-stroke-data.csv")


@dag(
    dag_id="etl_train_models_process_taskflow",
    description="Train the candidates, pick a finalist by out-of-fold F2, and gate it",
    schedule=None,
    start_date=datetime.datetime(2025, 1, 1),
    catchup=False,
    dagrun_timeout=datetime.timedelta(hours=1),
    default_args={
        "owner": "mlops-fiuba",
        "retries": 1,
        "retry_delay": datetime.timedelta(minutes=1),
    },
    params={
        "n_trials": Param(30, type="integer", minimum=1, description="Optuna trials per candidate"),
        "gate_min_p_better": Param(
            GATE_MIN_P_BETTER,
            type="number",
            minimum=0,
            maximum=1,
            description="Promote only if P(challenger's F2 > champion's) is at least this",
        ),
    },
    tags=["training", "gate"],
)
def process_etl_taskflow():
    @task
    def verify_raw() -> str:
        from stroke.data import data_version, verify_manifest

        return data_version(verify_manifest(Path(DATA_PATH)))

    @task
    def train_candidate(name: str, expected_version: str) -> dict:
        from model.train import train_candidate as train
        from stroke.data import frozen_data

        data = frozen_data(Path(DATA_PATH))
        if data.data_version != expected_version:  # every task must train on the same file
            raise RuntimeError(f"raw data changed: {expected_version} -> {data.data_version}")
        n_trials = int(get_current_context()["params"]["n_trials"])
        return train(name, data, n_trials=n_trials, pipeline="comparison-dag")

    # No retry: after a failure halfway, a retry would register a second version.
    @task(retries=0)
    def register_and_gate(summaries: list[dict]) -> dict:
        from model.train import register_and_gate as gate
        from stroke.data import frozen_data

        min_p_better = float(get_current_context()["params"]["gate_min_p_better"])
        return gate(summaries, frozen_data(Path(DATA_PATH)), min_p_better)

    version = verify_raw()
    summaries = [
        train_candidate.override(task_id=f"train_{name}")(name, version) for name in CANDIDATES
    ]
    register_and_gate(summaries)


dag = process_etl_taskflow()
