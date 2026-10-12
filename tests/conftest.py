"""Fixtures shared by every test file. Pytest loads this module automatically."""

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model.preprocess import build_feature_pipeline
from stroke.data import load_clean, split_train_test

CSV = Path(__file__).parent.parent / "data/raw/healthcare-dataset-stroke-data.csv"


@pytest.fixture(scope="session")
def split():
    X, y = load_clean(CSV)
    return split_train_test(X, y)  # X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def tiny_model(split):
    """A real fitted pipeline (about a second to train), so API tests never need MLflow."""
    X_train, _, y_train, _ = split
    model = Pipeline(
        [
            ("preprocess", build_feature_pipeline()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return model.fit(X_train, y_train)
