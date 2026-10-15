"""Raw data handling: load, clean, freeze the train/test split, and guard the raw file.

The raw CSV is immutable. `write_manifest` records its fingerprint once, when the file is
first stored; `verify_manifest` is meant to be called before any pipeline run, so a raw
file that changed underneath you fails loudly instead of silently shrinking the dataset.
That happened once: `dags/stroke_pipeline.py`'s `fetch_data` used to write a mutated
snapshot over this same raw file, and the dataset shrank from 5,110 to 2,093 rows over
four runs before anyone noticed.
"""

import hashlib
import io
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.preprocess import TARGET, load_data
from stroke.config import RANDOM_SEED, TEST_SIZE

log = logging.getLogger(__name__)

MANIFEST_NAME = "_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(csv_path: Path) -> dict:
    """Fingerprint the raw file next to itself. Run once, when the raw file is first stored."""
    df = pd.read_csv(csv_path)
    manifest = {
        "file": csv_path.name,
        "sha256": sha256_file(csv_path),
        "rows": len(df),
        "positives": int(df[TARGET].sum()),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (csv_path.parent / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(csv_path: Path) -> dict:
    """Raise if the raw file no longer matches its manifest. Returns the manifest."""
    manifest = json.loads((csv_path.parent / MANIFEST_NAME).read_text())
    actual = sha256_file(csv_path)
    if actual != manifest["sha256"]:
        raise ValueError(
            f"{csv_path.name} changed: expected {manifest['sha256'][:12]}, got {actual[:12]}"
        )
    return manifest


def data_version(manifest: dict) -> str:
    """Short id stamped on every MLflow run so a model can be traced to its exact data."""
    return manifest["sha256"][:12]


def _read_manifest_from_s3(csv_path: str) -> dict:
    """Read the manifest sitting next to an s3:// CSV (same bucket, same prefix).

    Mirrors model.preprocess._read_csv's S3 client setup so this works against MinIO
    the same way training's own data loading does.
    """
    import boto3
    from botocore.client import Config

    parsed = urlparse(csv_path)
    manifest_key = str(Path(parsed.path).parent / MANIFEST_NAME).lstrip("/")
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    obj = client.get_object(Bucket=parsed.netloc, Key=manifest_key)
    return json.loads(io.BytesIO(obj["Body"].read()).read())


def data_version_for(csv_path: str) -> str | None:
    """Best-effort data_version for whatever DATA_PATH training was pointed at.

    Works for a local path (data/raw/<file>.csv, with _MANIFEST.json beside it) or an
    s3://bucket/key URI (minio-init uploads _MANIFEST.json alongside the CSV, since it
    copies the whole data/raw/ directory). Returns None, rather than raising, when no
    manifest is found — tagging a model with its data version is best-effort, not a
    hard requirement, until every training path is guaranteed to have one.
    """
    try:
        if csv_path.startswith("s3://"):
            manifest = _read_manifest_from_s3(csv_path)
        else:
            manifest = json.loads((Path(csv_path).parent / MANIFEST_NAME).read_text())
        return data_version(manifest)
    except Exception:
        log.warning("no manifest found next to %s; logging without a data_version tag", csv_path)
        return None


def load_clean(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw CSV via model.preprocess.load_data (drops id, drops gender='Other')."""
    return load_data(str(csv_path))


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One stratified split, frozen by seed. The test rows are never used for tuning."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# The hospitals the service's simulated customers are: residence type x age 60 and over.
HOSPITALS = ("urban-60plus", "urban-under60", "rural-60plus", "rural-under60")


def assign_hospital(X: pd.DataFrame) -> pd.Series:
    """Which hospital each patient belongs to, from two raw features. The dataset has no
    hospital column, so this rule stands in for one. It is deterministic, so the event
    producer and the drift baselines always agree on who belongs where."""
    band = np.where(X["age"] >= 60, "-60plus", "-under60")
    return (X["Residence_type"].str.lower() + band).rename("hospital_id")
