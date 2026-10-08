"""
DAG: stroke_data_cleaning
Prepara los datos para el entrenamiento.

Tras unificar el preprocessing, este DAG ya NO encoda/imputa/escala: esa lógica
vive ahora en un único lugar (`model.preprocess.build_feature_pipeline`) y se
aplica dentro del propio modelo (Pipeline de sklearn). Acá solo se particiona el
dataset crudo y se persisten los splits en MinIO para que el DAG de comparación
los consuma.

Tareas
------
1. validate_source — verifica que el CSV de entrada existe y tiene el esquema esperado
2. split_and_upload — carga (drop id + gender='Other'), split estratificado 60/20/20
   y sube los 6 CSV crudos (X/y train/val/test) a MinIO en processed/final/
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.log.logging_mixin import LoggingMixin
from botocore.client import Config
from sklearn.model_selection import train_test_split

from model.preprocess import load_data

DATA_PATH = Path(os.getenv("DATA_PATH", "/opt/airflow/data/healthcare-dataset-stroke-data.csv"))
BUCKET = "mlflow-artifacts"
# Mismo prefijo que lee el DAG de comparación (DIR_DATA_PROCESSED = "processed/final")
FINAL_PREFIX = "processed/final"

MINIO_URL = os.environ["MLFLOW_S3_ENDPOINT_URL"]
MINIO_ROOT_USER = os.environ["AWS_ACCESS_KEY_ID"]
MINIO_ROOT_PASSWORD = os.environ["AWS_SECRET_ACCESS_KEY"]

logger = LoggingMixin().log

default_args = {
    "owner": "mlops-fiuba",
    "retries": 0,
    "retry_delay": timedelta(minutes=0.5),
}


# helpers
def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _df_to_s3(df: pd.DataFrame, s3_client, key: str) -> None:
    """Sube un DataFrame como CSV a S3/MinIO."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    logger.info("Subido %s (%d filas, %d cols)", key, len(df), df.shape[1])


# Tarea 1: validate_source
def validate_source(**context) -> None:
    """Verifica que el CSV de entrada exista y tenga las columnas esperadas."""
    data_path = context["params"].get("data_path", str(DATA_PATH))

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    df = pd.read_csv(data_path, nrows=5)
    required = {
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
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en el CSV: {missing_cols}")

    logger.info("Validación OK — path=%s", data_path)
    context["ti"].xcom_push(key="data_path", value=str(data_path))


# Tarea 2: split_and_upload
def split_and_upload(**context) -> None:
    """
    Carga el CSV crudo con `model.preprocess.load_data` (misma carga que usa el
    entrenamiento: drop de 'id' y de filas gender='Other'), hace un split
    estratificado 60/20/20 y sube los 6 CSV CRUDOS a MinIO.

    No se aplica imputación/encoding/escalado: eso lo hace el Pipeline del modelo
    en el DAG de comparación, garantizando una única lógica de preprocessing.
    """
    ti = context["ti"]
    data_path = ti.xcom_pull(task_ids="validate_source", key="data_path") or str(DATA_PATH)
    s3 = _get_s3_client()

    # Carga unificada (idéntica a model/train.py)
    X, y = load_data(data_path)
    logger.info("Dataset cargado: X=%s  y=%s", X.shape, y.shape)

    # Split 60/20/20 estratificado
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    logger.info("Split — train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # Subir los 6 splits crudos a processed/final/
    for name, frame in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
        ("y_train", y_train.to_frame()),
        ("y_val", y_val.to_frame()),
        ("y_test", y_test.to_frame()),
    ]:
        _df_to_s3(frame, s3, f"{FINAL_PREFIX}/{name}.csv")

    logger.info("Splits crudos disponibles en s3://%s/%s/", BUCKET, FINAL_PREFIX)


# DAG
with DAG(
    dag_id="stroke_data_cleaning",
    description="Particiona el dataset crudo y sube los splits a MinIO (sin preprocessing)",
    schedule_interval=None,  # disparo manual o desde el orquestador
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "preprocessing", "stroke"],
    params={"data_path": str(DATA_PATH)},
    doc_md=__doc__,
) as dag:
    t1_validate = PythonOperator(
        task_id="validate_source",
        python_callable=validate_source,
    )

    t2_split = PythonOperator(
        task_id="split_and_upload",
        python_callable=split_and_upload,
    )

    t1_validate >> t2_split
