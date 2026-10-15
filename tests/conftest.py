"""Fixtures shared by every test file. Pytest loads this module automatically."""

import time
from dataclasses import dataclass
from pathlib import Path

import grpc
import pytest
from grpc_health.v1 import health
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model.preprocess import build_feature_pipeline
from services.model_server.holder import ModelHolder, ModelState
from services.model_server.main import create_server
from stroke.data import load_clean, split_train_test
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

CSV = Path(__file__).parent.parent / "data/raw/healthcare-dataset-stroke-data.csv"


@pytest.fixture(scope="session")
def raw_csv() -> Path:
    """The immutable source dataset. Tests may read it, never write it."""
    return CSV


@pytest.fixture(scope="session")
def dataset():
    """(X, y) after load_clean. Shared by every test, so treat it as read-only."""
    return load_clean(CSV)


@pytest.fixture(scope="session")
def split(dataset):
    X, y = dataset
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


@pytest.fixture
def make_state(tiny_model):
    """Factory for the state the model server holds, wrapping `tiny_model` by default."""

    def make(version: str = "1", threshold: float = 0.5, model=None) -> ModelState:
        return ModelState(
            model=tiny_model if model is None else model,
            name="stroke-model",
            version=version,
            threshold=threshold,
            family="logreg",
            data_version="abc123",
            loaded_at=time.time(),
        )

    return make


@dataclass
class RunningServer:
    holder: ModelHolder
    health: health.HealthServicer
    target: str


@pytest.fixture
def model_server():
    """A real gRPC model server on a free local port, started with NO model loaded.
    Tests put a model in with `model_server.holder.set(make_state())`."""
    holder = ModelHolder()
    server, health_servicer, port = create_server(holder, "localhost:0")  # 0: any free port
    server.start()
    yield RunningServer(holder, health_servicer, f"localhost:{port}")
    server.stop(None)


@pytest.fixture
def stub(model_server):
    with grpc.insecure_channel(model_server.target) as channel:
        yield pb_grpc.ScoringStub(channel)
