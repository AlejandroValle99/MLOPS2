import io
import os

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config

DATASET_BUCKET = os.getenv("DATASET_BUCKET", "datasets")
DATASET_KEY = os.getenv("DATASET_KEY", "healthcare-dataset-stroke-data.csv")
MUTATION_SAMPLE_RATE = float(os.getenv("MUTATION_SAMPLE_RATE", "0.80"))
MUTATION_NOISE_STD = float(os.getenv("MUTATION_NOISE_STD", "0.03"))


_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (0.0, 100.0),
    "avg_glucose_level": (50.0, 300.0),
    "bmi": (10.0, 70.0),
}


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def fetch_dataset() -> pd.DataFrame:
    client = _get_s3_client()
    obj = client.get_object(Bucket=DATASET_BUCKET, Key=DATASET_KEY)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def apply_mutations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(frac=MUTATION_SAMPLE_RATE).copy()

    for col, (low, high) in _NUMERIC_BOUNDS.items():
        noise = np.random.normal(0, MUTATION_NOISE_STD * df[col].std(), size=len(df))
        df[col] = (df[col] + noise).clip(low, high)

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    return df.sample(frac=1).reset_index(drop=True)
