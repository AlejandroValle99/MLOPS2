"""DAG de Airflow — pipeline de entrenamiento del modelo de stroke."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://api:8000")

# Fixed bug (MLOps 1 -> MLOps 2 move, Phase 0): fetch_data used to write the mutated
# /dataset response over this same path, which is the bind-mounted RAW source file.
# minio-init re-uploaded that mutated copy on every restart, so the dataset shrank
# 5,110 -> 4,088 -> 3,270 -> 2,616 -> 2,093 rows across four runs before anyone noticed.
# RAW_DATA_PATH is read-only here, never written to. fetch_data writes to SNAPSHOT_PATH
# instead, and validate_data/train_model read the snapshot, not the raw file.
RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH", "/opt/airflow/data/raw/healthcare-dataset-stroke-data.csv"
)
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH", "/opt/airflow/data/snapshots/stroke_snapshot.csv")

default_args = {
    "owner": "mlops-fiuba",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


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


def train_model() -> None:
    """Entrena el Random Forest sobre el snapshot y loggea el run a MLflow."""
    import os

    # model.train reads DATA_PATH at import time; set it before the lazy import below
    # so this task trains on the snapshot fetch_data wrote, not on the raw file.
    os.environ["DATA_PATH"] = os.getenv("SNAPSHOT_PATH", SNAPSHOT_PATH)
    from model.train import train

    train()


with DAG(
    dag_id="stroke_prediction_pipeline",
    default_args=default_args,
    description="Pipeline de entrenamiento del modelo de prediccion de stroke",
    # schedule=timedelta(weeks=1),
    schedule=None,  # manual
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["stroke", "mlops", "training"],
) as dag:
    task_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_fetch >> task_validate >> task_train
