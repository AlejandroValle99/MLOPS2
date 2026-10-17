"""Raw data handling: load, clean, freeze the train/test split, and guard the raw file.

Training loads its data through `frozen_data` (or `snapshot_data` for a retrain on a fresh
snapshot), which checks the raw file first and always tests on the same frozen rows.

The raw CSV is immutable. `write_manifest` records its fingerprint once, when the file is
first stored; `verify_manifest` is meant to be called before any pipeline run, so a raw
file that changed underneath you fails loudly instead of silently shrinking the dataset.
That happened once: `dags/stroke_pipeline.py`'s `fetch_data` used to write a mutated
snapshot over this same raw file, and the dataset shrank from 5,110 to 2,093 rows over
four runs before anyone noticed.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.preprocess import TARGET, load_data, split_features_target
from stroke.config import RANDOM_SEED, TEST_SIZE

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


def load_clean(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw CSV via model.preprocess.load_data (drops id, drops gender='Other')."""
    return load_data(str(csv_path))


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One stratified split, frozen by seed. The test rows are never used for tuning."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def frozen_split_id(manifest: dict, seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> str:
    """Names the frozen test set: which raw file, which seed, what size. Every version the
    gate registers carries it, and the gate only compares two models whose ids match."""
    return f"{data_version(manifest)}-seed{seed}-test{test_size:g}"


@dataclass(frozen=True)
class TrainingData:
    """What one training run uses. The test split is always the frozen one from the raw
    file; the training rows are usually its training split, or a fresh snapshot's rows."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    data_version: str  # the training rows' data
    test_split: str  # the test set's identity (frozen_split_id)


def frozen_data(raw_csv: Path) -> TrainingData:
    """The frozen split of the raw file, after checking the file against its manifest."""
    manifest = verify_manifest(raw_csv)
    X, y = load_clean(raw_csv)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    return TrainingData(
        X_train, y_train, X_test, y_test, data_version(manifest), frozen_split_id(manifest)
    )


def exclude_frozen_test(snapshot: pd.DataFrame, raw_csv: Path) -> pd.DataFrame:
    """A snapshot's rows minus every patient in the frozen test set, matched by `id`.

    Snapshots are resampled from the same patients as the raw file, with noise added, so
    without this a model trained on one would have seen copies of the very rows the gate
    then judges it on.
    """
    raw_ids = pd.read_csv(raw_csv, usecols=["id"])["id"]
    X, y = load_clean(raw_csv)  # keeps the raw file's row labels, so they map back to ids
    _, X_test, _, _ = split_train_test(X, y)
    return snapshot[~snapshot["id"].isin(raw_ids.loc[X_test.index])]


# Clipping bounds for the noise apply_mutations adds, so a value never leaves its range.
_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (0.0, 100.0),
    "avg_glucose_level": (50.0, 300.0),
    "bmi": (10.0, 70.0),
}


def apply_mutations(
    df: pd.DataFrame, sample_rate: float = 0.80, noise_std: float = 0.03
) -> pd.DataFrame:
    """A stand-in for newly collected data: a random `sample_rate` of the raw rows, with
    Gaussian noise of `noise_std` standard deviations on each numeric feature. The retrain
    DAG builds its snapshot with it. Keeps the `id` column, so exclude_frozen_test can still
    drop every frozen test patient."""
    df = df.sample(frac=sample_rate).copy()

    for col, (low, high) in _NUMERIC_BOUNDS.items():
        noise = np.random.normal(0, noise_std * df[col].std(), size=len(df))
        df[col] = (df[col] + noise).clip(low, high)

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    return df.sample(frac=1).reset_index(drop=True)


def snapshot_data(snapshot_csv: Path, raw_csv: Path) -> TrainingData:
    """Train on a fresh snapshot, test on the raw file's frozen test set. The model is then
    judged on the same rows as every other version, none of which it trained on."""
    frozen = frozen_data(raw_csv)
    X_train, y_train = split_features_target(
        exclude_frozen_test(pd.read_csv(snapshot_csv), raw_csv)
    )
    return replace(
        frozen, X_train=X_train, y_train=y_train, data_version=sha256_file(snapshot_csv)[:12]
    )


# The hospitals the service's simulated customers are: residence type x age 60 and over.
HOSPITALS = ("urban-60plus", "urban-under60", "rural-60plus", "rural-under60")


def assign_hospital(X: pd.DataFrame) -> pd.Series:
    """Which hospital each patient belongs to, from two raw features. The dataset has no
    hospital column, so this rule stands in for one. It is deterministic, so the event
    producer and the drift baselines always agree on who belongs where."""
    band = np.where(X["age"] >= 60, "-60plus", "-under60")
    return (X["Residence_type"].str.lower() + band).rename("hospital_id")
