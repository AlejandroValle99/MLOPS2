"""Regression tests for Phase 0 (docs/PLAN.md): the move from TP_MLops and the bug fixes
applied during it. Run with: uv run --group ml pytest tests/test_phase0_move_and_fix.py
"""

from pathlib import Path

import pandas as pd
import pytest

from model.preprocess import RAW_COLUMNS, SMOKING_STATUSES, WORK_TYPES, build_feature_pipeline
from stroke.data import data_version, load_clean, split_train_test, verify_manifest, write_manifest

CSV = Path(__file__).parent.parent / "data/raw/healthcare-dataset-stroke-data.csv"


@pytest.fixture(scope="module")
def manifest():
    """The committed manifest, checked against the CSV. Never write_manifest() here: that
    would rewrite data/raw/ on every run and re-fingerprint a corrupted file as the reference."""
    return verify_manifest(CSV)


def test_the_restored_dataset_is_the_original_not_the_corrupted_head(manifest):
    """TP_MLops HEAD had 2,093 rows; the original (git commit 82646ef) has 5,110."""
    assert manifest["rows"] == 5110
    assert manifest["positives"] == 249


def test_verify_manifest_passes_on_the_untouched_file(manifest):
    verify_manifest(CSV)  # must not raise


def test_verify_manifest_catches_the_exact_mlops1_corruption(manifest, tmp_path):
    """The bug: a DAG overwrote the raw file with an 80%-sampled, noised copy."""
    tampered = tmp_path / CSV.name
    tampered.write_bytes(CSV.read_bytes())
    write_manifest(tampered)
    df = pd.read_csv(tampered).sample(frac=0.8, random_state=0)
    df.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(tampered)


def test_data_version_is_a_short_stable_id(manifest):
    assert data_version(manifest) == manifest["sha256"][:12]


def test_load_clean_drops_id_and_the_single_other_gender_row():
    X, y = load_clean(CSV)
    assert X.shape == (5109, 10)
    assert int(y.sum()) == 249
    assert list(X.columns) == RAW_COLUMNS
    assert "Other" not in X["gender"].unique()


def test_split_is_deterministic_and_keeps_the_positive_ratio():
    X, y = load_clean(CSV)
    X_tr1, X_te1, y_tr1, y_te1 = split_train_test(X, y)
    X_tr2, _, y_tr2, _ = split_train_test(X, y)
    assert X_tr1.index.equals(X_tr2.index)  # same seed -> same rows
    assert len(X_tr1) == 4087 and int(y_tr1.sum()) == 199
    assert len(X_te1) == 1022 and int(y_te1.sum()) == 50


def test_feature_pipeline_output_has_14_columns():
    X, y = load_clean(CSV)
    import numpy as np

    Xt = build_feature_pipeline().fit_transform(X, y)
    assert Xt.shape[1] == 14
    assert not np.isnan(Xt).any()


def test_feature_pipeline_width_is_stable_when_a_partition_lacks_categories():
    """The bug: OneHotEncoder inferred categories from the training split, so a split
    missing a category produced 11 columns instead of 14 — breaking federation, where
    each client's local split is small and may not contain every category."""
    X, y = load_clean(CSV)
    partial = X[X.work_type.isin(["Private", "children"])]
    pipe = build_feature_pipeline().fit(partial)
    assert pipe.transform(X.head(5)).shape[1] == 14


def test_category_lists_cover_every_value_in_the_real_data():
    X, _ = load_clean(CSV)
    assert set(X["work_type"].unique()) <= set(WORK_TYPES)
    cleaned_smoking = X["smoking_status"].replace("Unknown", "never smoked")
    assert set(cleaned_smoking.unique()) <= set(SMOKING_STATUSES)
