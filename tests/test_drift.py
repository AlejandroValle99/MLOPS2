"""stroke.drift: baselines, PSI and the drift report, plus the committed baseline file."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_drift_baseline import OUT, build
from stroke.drift import MONITORED, build_reference, drift_report, psi


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_psi_grows_with_the_shift_and_survives_an_empty_bin():
    small = psi([0.5, 0.5], [0.55, 0.45])
    large = psi([0.5, 0.5], [0.9, 0.1])
    assert 0 < small < large
    assert math.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))  # ln(0) would be -inf


def test_a_reference_is_plain_json_with_decile_bins(split):
    X_train, _, _, _ = split
    reference = build_reference(X_train)
    assert json.loads(json.dumps(reference)) == reference
    glucose = reference["numeric"]["avg_glucose_level"]
    assert len(glucose["inner_edges"]) == 9
    assert sum(glucose["proportions"]) == pytest.approx(1)
    assert set(reference["numeric"]) | set(reference["categorical"]) == set(MONITORED)


def test_psi_is_low_on_the_same_population_and_high_after_a_glucose_shift(split):
    X_train, X_test, _, _ = split
    reference = build_reference(X_train)
    assert drift_report(reference, X_test, min_count=300)["avg_glucose_level"] < 0.1
    shifted = X_test.assign(avg_glucose_level=X_test["avg_glucose_level"] * 1.2)
    assert drift_report(reference, shifted, min_count=300)["avg_glucose_level"] > 0.25


def test_a_window_below_the_minimum_reports_nothing(split):
    X_train, X_test, _, _ = split
    assert drift_report(build_reference(X_train), X_test.head(299), min_count=300) == {}


def test_a_category_the_baseline_never_saw_counts_as_drift(split):
    X_train, X_test, _, _ = split
    reference = build_reference(X_train)
    window = X_test.head(300).assign(work_type="Astronaut")
    assert drift_report(reference, window, min_count=300)["work_type"] > 0.25


def test_values_outside_the_baseline_range_land_in_the_outer_bins(split):
    X_train, _, _, _ = split
    reference = build_reference(X_train)
    window = pd.DataFrame({col: X_train[col].head(300) for col in MONITORED})
    window["age"] = np.where(np.arange(300) % 2 == 0, 500.0, -5.0)  # all off the scale
    assert drift_report(reference, window, min_count=300)["age"] > 1  # counted, not dropped


def test_the_committed_baseline_file_matches_the_frozen_split():
    """data/drift/reference_stats.json is generated, then committed so the worker's image
    needs no scikit-learn. Regenerate it with: python -m scripts.build_drift_baseline"""
    committed = json.loads(Path(OUT).read_text())
    assert committed == json.loads(json.dumps(build())), (
        "the drift baseline is stale; run: uv run --group ml python -m scripts.build_drift_baseline"
    )
