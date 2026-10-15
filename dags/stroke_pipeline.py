"""DAG stroke_prediction_pipeline: retrain on a fresh snapshot, then gate the result.

    verify_raw -> fetch_data -> validate_data -> train_model

The snapshot stands in for newly collected data: the same patients resampled, with noise.
train_model trains every candidate on it, minus the frozen test patients, and gates the
finalist against the champion on the frozen test set from the raw file. So a retrained
model is judged on the same rows as every other version, none of which it trained on.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from stroke.config import GATE_MIN_P_BETTER

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://api:8000")

# fetch_data used to write the mutated /dataset response over this same path, which is
# the bind-mounted RAW source file. minio-init re-uploaded that mutated copy on every
# restart, so the dataset shrank 5,110 -> 4,088 -> 3,270 -> 2,616 -> 2,093 rows across
# four runs before anyone noticed. RAW_DATA_PATH is read-only here, never written to:
# fetch_data writes to SNAPSHOT_PATH instead. train_model trains on the snapshot and reads
# the raw file only for the frozen test set.
RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH", "/opt/airflow/data/raw/healthcare-dataset-stroke-data.csv"
)
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH", "/opt/airflow/data/snapshots/stroke_snapshot.csv")

default_args = {
    "owner": "mlops-fiuba",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def verify_raw() -> str:
    """Fails the run if the raw file no longer matches its manifest; returns its version."""
    from stroke.data import data_version, verify_manifest

    return data_version(verify_manifest(Path(RAW_DATA_PATH)))


def fetch_data() -> None:
    """Downloads a fresh dataset snapshot from the API into SNAPSHOT_PATH.

    Never writes to RAW_DATA_PATH — see the note above the module-level constants.
    """
    import urllib.request

    snapshot_path = Path(SNAPSHOT_PATH)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATA_SERVICE_URL}/dataset"

    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"/dataset returned HTTP {response.status}")
        content = response.read()

    snapshot_path.write_bytes(content)
    print(f"Dataset escrito en {snapshot_path} ({len(content)} bytes)")


def validate_data() -> None:
    """Verifica que el snapshot exista y tenga el esquema esperado."""
    import os

    import pandas as pd

    data_path = Path(os.getenv("SNAPSHOT_PATH", SNAPSHOT_PATH))

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    df = pd.read_csv(data_path)

    expected_cols = {
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    if len(df) < 100:
        raise ValueError(f"Dataset demasiado pequeño: {len(df)} filas")

    print(f"Dataset validado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Stroke rate: {df['stroke'].mean():.2%}")


def train_model(**context) -> dict:
    """Trains every candidate on the snapshot and sends the finalist through the gate."""
    from model.train import train_and_gate
    from stroke.data import snapshot_data

    params = context["params"]
    return train_and_gate(
        snapshot_data(Path(SNAPSHOT_PATH), Path(RAW_DATA_PATH)),
        n_trials=int(params["n_trials"]),
        pipeline="retrain-dag",
        min_p_better=float(params["gate_min_p_better"]),
    )


with DAG(
    dag_id="stroke_prediction_pipeline",
    default_args=default_args,
    description="Retrain on a fresh snapshot and gate the finalist",
    # schedule=timedelta(weeks=1),
    schedule=None,  # manual
    start_date=datetime(2025, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(hours=1),
    params={
        "n_trials": Param(30, type="integer", minimum=1, description="Optuna trials per candidate"),
        "gate_min_p_better": Param(GATE_MIN_P_BETTER, type="number", minimum=0, maximum=1),
    },
    tags=["stroke", "mlops", "training"],
) as dag:
    task_verify = PythonOperator(task_id="verify_raw", python_callable=verify_raw)

    task_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        retries=0,  # after a failure halfway, a retry would register a second version
    )

    task_verify >> task_fetch >> task_validate >> task_train
