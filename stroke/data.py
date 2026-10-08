"""Raw data handling: load, clean, freeze the train/test split, and guard the raw file.

The raw CSV is immutable. `write_manifest` records its fingerprint once, when the file is
first stored; `verify_manifest` is meant to be called before any pipeline run, so a raw
file that changed underneath you fails loudly instead of silently shrinking the dataset —
this is the manifest for the exact bug that corrupted the MLOps 1 dataset (see
Appendix C / Phase 0 of docs/PLAN.md): `dags/stroke_pipeline.py`'s `fetch_data` used to
write a mutated snapshot over this same raw file.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from model.preprocess import TARGET, load_data
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
