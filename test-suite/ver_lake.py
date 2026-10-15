"""Muestra la zona outputs del lake: cuántos archivos y filas hay por día, y las últimas
filas de cada tabla, con sus columnas.

    docker compose --env-file .env.local exec -T lake-sink python - < test-suite/ver_lake.py

Corre dentro del contenedor lake-sink, que ya tiene boto3, pyarrow y las credenciales de
MinIO: no hace falta nada instalado ni configurado en la laptop.
"""

import io
import os
from collections import Counter

import boto3
import pyarrow.parquet as pq

FEATURES = {
    "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
    "residence_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
}  # fmt: skip
SHOW = {
    "predictions": ["hospital_id", "patient_id", "probability", "high_risk", "model_version"],
    "window-reports": ["hospital_id", "count", "positive_rate", "psi_avg_glucose_level", "alerts"],
}

s3 = boto3.client("s3")
bucket = os.environ["LAKE_BUCKET"]


def keys(prefix: str) -> list[str]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out += [obj["Key"] for obj in page.get("Contents", [])]
    return out


def table(key: str):
    return pq.read_table(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))


def fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


print(f"s3://{bucket}/outputs/  (endpoint {s3.meta.endpoint_url})")
for topic, columns in SHOW.items():
    found = keys(f"outputs/{topic}/")
    print(f"\n== {topic}: {len(found)} archivos")
    if not found:
        continue
    rows_per_day: Counter = Counter()
    for key in found:
        rows_per_day[key.split("/")[2]] += pq.ParquetFile(
            io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        ).metadata.num_rows
    for day, rows in sorted(rows_per_day.items()):
        print(f"   {day}: {rows} filas")
    last = max(found, key=lambda k: (k.rsplit("/", 1)[1], k))  # the highest offsets
    t = table(last)
    print(f"   último archivo: {last}  ({t.num_rows} filas)")
    print(f"   columnas: {', '.join(t.column_names)}")
    leaked = sorted(FEATURES & set(t.column_names))
    print(f"   columnas con datos del paciente: {leaked or 'ninguna'}")
    print("   últimas filas:")
    for row in t.slice(max(t.num_rows - 3, 0)).select(columns).to_pylist():
        print("     " + "  ".join(f"{col}={fmt(row[col])}" for col in columns))
