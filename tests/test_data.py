"""stroke.data: the raw file's integrity manifest, loading, and the frozen train/test split."""

import pandas as pd
import pytest

from model.preprocess import RAW_COLUMNS
from stroke.data import data_version, split_train_test, verify_manifest, write_manifest


@pytest.fixture(scope="module")
def manifest(raw_csv):
    """The committed manifest, checked against the CSV. Never write_manifest() here: that
    would rewrite data/raw/ on every run and re-fingerprint a corrupted file as the reference."""
    return verify_manifest(raw_csv)


def test_raw_dataset_has_the_expected_size(manifest):
    assert manifest["rows"] == 5110
    assert manifest["positives"] == 249


def test_verify_manifest_passes_on_the_untouched_file(raw_csv):
    verify_manifest(raw_csv)  # must not raise


def test_verify_manifest_detects_a_modified_file(raw_csv, tmp_path):
    """The failure the manifest guards against: a pipeline overwriting the raw file with a
    smaller, resampled copy of itself."""
    copy = tmp_path / raw_csv.name
    copy.write_bytes(raw_csv.read_bytes())
    write_manifest(copy)
    pd.read_csv(copy).sample(frac=0.8, random_state=0).to_csv(copy, index=False)
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(copy)


def test_data_version_is_a_short_stable_id(manifest):
    assert data_version(manifest) == manifest["sha256"][:12]


def test_load_clean_drops_id_and_the_single_other_gender_row(dataset):
    X, y = dataset
    assert X.shape == (5109, 10)
    assert int(y.sum()) == 249
    assert list(X.columns) == RAW_COLUMNS
    assert "Other" not in X["gender"].unique()


def test_split_is_deterministic_and_keeps_the_positive_ratio(dataset):
    X, y = dataset
    X_tr1, X_te1, y_tr1, y_te1 = split_train_test(X, y)
    X_tr2, _, _, _ = split_train_test(X, y)
    assert X_tr1.index.equals(X_tr2.index)  # same seed -> same rows
    assert len(X_tr1) == 4087 and int(y_tr1.sum()) == 199
    assert len(X_te1) == 1022 and int(y_te1.sum()) == 50
