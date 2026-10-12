import io
import os
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "stroke"

# Public so other modules (stroke_rpc.convert, stroke.federated, tests) can import the
# exact raw feature names instead of hand-copying them.
RAW_COLUMNS = [
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
]

_NUM_COLS = ["age", "avg_glucose_level", "bmi"]
_BIN_COLS = [
    "hypertension",
    "heart_disease",
    "gender",
    "ever_married",
    "Residence_type",
]
_WORK_COL = ["work_type"]
_SMOKE_COL = ["smoking_status"]

# Fixed, exhaustive category lists for the one-hot columns. Inferring the categories from
# the training data instead would make a split that lacks a category produce fewer output
# columns (11 instead of 14). Pinning categories= makes every fit produce the same
# 14-column output regardless of what the training data happens to contain, which
# federated learning needs: each client's local split is small and may not include every
# category.
WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
# "Unknown" smoking status is remapped to "never smoked" by StrokeCleaner before this
# encoder ever sees the column, so "Unknown" is deliberately not listed here.
SMOKING_STATUSES = ["never smoked", "formerly smoked", "smokes"]

_AGE_BINS = [0, 10, 20, 30, 70, np.inf]
_AGE_LABELS = ["0-10", "11-20", "21-30", "31-70", "71+"]


class StrokeCleaner(BaseEstimator, TransformerMixin):
    """Applies deterministic cleaning and binary encoding to the raw stroke DataFrame."""

    def fit(self, X: pd.DataFrame, y=None) -> "StrokeCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # "Unknown" smoking maps to "never smoked" per notebook 01 decision
        df["smoking_status"] = df["smoking_status"].replace("Unknown", "never smoked")
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)
        df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0}).astype(int)
        df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0}).astype(int)
        return df


class BMIGroupImputer(BaseEstimator, TransformerMixin):
    """Imputes missing BMI using age-group medians fitted only on training data."""

    def fit(self, X: pd.DataFrame, y=None) -> "BMIGroupImputer":
        df = X.copy()
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["_age_group"] = pd.cut(
            df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, include_lowest=True
        )
        medians = df.groupby("_age_group", observed=False)["bmi"].median()
        self.group_medians_: dict[str, float] = {str(k): float(v) for k, v in medians.items()}
        self.global_median_: float = float(df["bmi"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["_age_group"] = pd.cut(
            df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, include_lowest=True
        )
        fill = df["_age_group"].astype(str).map(self.group_medians_).astype(float)
        df["bmi"] = df["bmi"].fillna(fill.fillna(self.global_median_))
        return df.drop(columns=["_age_group"])


def _build_column_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), _NUM_COLS),
            ("pass", "passthrough", _BIN_COLS),
            (
                "work",
                OneHotEncoder(
                    categories=[WORK_TYPES],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                _WORK_COL,
            ),
            (
                "smoke",
                OneHotEncoder(
                    categories=[SMOKING_STATUSES],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                _SMOKE_COL,
            ),
        ],
        remainder="drop",
    )


def build_feature_pipeline() -> Pipeline:
    """Returns an unfitted sklearn Pipeline that replicates the notebook 01 preprocessing."""
    return Pipeline(
        [
            ("cleaner", StrokeCleaner()),
            ("bmi_imputer", BMIGroupImputer()),
            ("features", _build_column_transformer()),
        ]
    )


def _read_csv(csv_path: str) -> pd.DataFrame:
    if not csv_path.startswith("s3://"):
        return pd.read_csv(csv_path)
    parsed = urlparse(csv_path)
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    obj = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads raw CSV, drops id and gender='Other' rows, returns (X, y)."""
    df = _read_csv(csv_path)
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"]
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y
