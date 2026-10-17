"""stroke.data: the raw file's integrity manifest, loading, and the frozen train/test split."""

import pandas as pd
import pytest

from model.preprocess import RAW_COLUMNS
from stroke.data import (
    HOSPITALS,
    apply_mutations,
    assign_hospital,
    data_version,
    exclude_frozen_test,
    frozen_data,
    load_clean,
    sha256_file,
    snapshot_data,
    split_train_test,
    verify_manifest,
    write_manifest,
)


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


def test_frozen_data_is_the_verified_frozen_split(raw_csv, split, manifest):
    data = frozen_data(raw_csv)
    X_train, X_test, _, _ = split
    assert data.X_train.index.equals(X_train.index)
    assert data.X_test.index.equals(X_test.index)
    assert data.data_version == data_version(manifest)
    assert data.test_split == f"{data_version(manifest)}-seed42-test0.2"


def test_frozen_data_refuses_a_modified_raw_file(raw_csv, tmp_path):
    copy = tmp_path / raw_csv.name
    copy.write_bytes(raw_csv.read_bytes())
    write_manifest(copy)
    copy.write_bytes(copy.read_bytes().replace(b"Urban", b"Rural", 1))  # one patient moves
    with pytest.raises(ValueError, match="changed"):
        frozen_data(copy)


@pytest.fixture(scope="module")
def frozen_test_ids(raw_csv):
    """The ids of the frozen test patients, worked out independently of stroke.data."""
    raw = pd.read_csv(raw_csv)
    X, y = load_clean(raw_csv)
    _, X_test, _, _ = split_train_test(X, y)
    assert raw.loc[X_test.index, "age"].equals(X_test["age"])  # row labels map back to raw
    return raw.loc[X_test.index, "id"]


def test_a_snapshot_never_trains_on_a_frozen_test_patient(raw_csv, frozen_test_ids):
    """A snapshot is the same patients resampled with noise (what the retrain DAG fetches),
    so it contains test patients. None of them may reach a training set."""
    snapshot = apply_mutations(pd.read_csv(raw_csv))
    assert snapshot["id"].isin(frozen_test_ids).sum() > 100  # the problem is real
    kept = exclude_frozen_test(snapshot, raw_csv)
    assert not kept["id"].isin(frozen_test_ids).any()
    assert len(kept) == (~snapshot["id"].isin(frozen_test_ids)).sum()  # nothing else dropped


def test_snapshot_data_trains_on_the_snapshot_and_tests_on_the_frozen_split(
    raw_csv, tmp_path, frozen_test_ids
):
    snapshot = tmp_path / "snapshot.csv"
    apply_mutations(pd.read_csv(raw_csv)).to_csv(snapshot, index=False)
    data, frozen = snapshot_data(snapshot, raw_csv), frozen_data(raw_csv)
    assert data.X_test.equals(frozen.X_test) and data.y_test.equals(frozen.y_test)
    assert data.test_split == frozen.test_split  # so the gate compares it with the champion
    assert data.data_version == sha256_file(snapshot)[:12]
    assert list(data.X_train.columns) == RAW_COLUMNS
    assert 0 < len(data.X_train) < len(frozen.X_train)  # 80% of patients, test ones removed


def test_every_patient_belongs_to_exactly_one_of_the_four_hospitals(dataset):
    X, _ = dataset
    hospitals = assign_hospital(X)
    assert hospitals.index.equals(X.index)
    assert set(hospitals.unique()) == set(HOSPITALS)
    assert hospitals.value_counts().to_dict() == {
        "urban-under60": 1889,
        "rural-under60": 1844,
        "urban-60plus": 707,
        "rural-60plus": 669,
    }


def test_the_hospital_rule_reads_residence_and_age_60_and_over():
    X = pd.DataFrame({"Residence_type": ["Urban", "Rural", "Rural"], "age": [60.0, 59.9, 82.0]})
    assert assign_hospital(X).tolist() == ["urban-60plus", "rural-under60", "rural-60plus"]
