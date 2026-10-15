"""Writes the drift baselines the stream worker compares each hospital's windows with.

    uv run --group ml python -m scripts.build_drift_baseline

One baseline per hospital, built from that hospital's rows of the frozen training split.
The output, data/drift/reference_stats.json, is committed so the worker's image needs no
scikit-learn; tests/test_drift.py fails if the file no longer matches what this script
would write (after a change to the split, the cleaning or the drift module).
"""

import json
from pathlib import Path

from stroke.data import assign_hospital, data_version, load_clean, split_train_test, verify_manifest
from stroke.drift import reference_document

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data/raw/healthcare-dataset-stroke-data.csv"
OUT = ROOT / "data/drift/reference_stats.json"


def build() -> dict:
    X, y = load_clean(RAW_CSV)
    X_train, _, _, _ = split_train_test(X, y)
    version = data_version(verify_manifest(RAW_CSV))
    return reference_document(X_train, assign_hospital(X_train), version)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
