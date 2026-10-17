"""Fixtures shared by every test file. Pytest loads this module automatically."""

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pytest
from grpc_health.v1 import health
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model.preprocess import build_feature_pipeline
from services.model_server.holder import ModelHolder, ModelState
from services.model_server.main import create_server
from stroke.config import EXPERIMENT_NAME
from stroke.data import load_clean, split_train_test
from stroke_rpc.channel import open_channel
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

ROOT = Path(__file__).parent.parent
CSV = ROOT / "data/raw/healthcare-dataset-stroke-data.csv"

# The key the test model server accepts; clients of it send this one.
UPSTREAM_KEY = "test-upstream-key"


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
    key: str  # what a client must send on every call but health


@pytest.fixture
def model_server():
    """A real gRPC model server on a free local port (plaintext, keyed with UPSTREAM_KEY),
    started with NO model loaded. Tests put a model in with
    `model_server.holder.set(make_state())`."""
    holder = ModelHolder()
    # 0: any free port
    server, health_servicer, port = create_server(holder, "localhost:0", keys={UPSTREAM_KEY})
    server.start()
    yield RunningServer(holder, health_servicer, f"localhost:{port}", UPSTREAM_KEY)
    server.stop(None)


@pytest.fixture
def stub(model_server):
    with open_channel(model_server.target, key=model_server.key) as channel:
        yield pb_grpc.ScoringStub(channel)


@pytest.fixture(scope="session")
def certs(tmp_path_factory) -> Path:
    """A dev CA, a server certificate for localhost and a client certificate, made by the
    project's own scripts/make_certs.sh in a temporary folder (never in certs/)."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    out = tmp_path_factory.mktemp("certs")
    subprocess.run(
        ["bash", str(ROOT / "scripts/make_certs.sh"), str(out)], check=True, capture_output=True
    )
    return out


@pytest.fixture(scope="session")
def mlflow_store(tmp_path_factory):
    """A throwaway MLflow for the whole session: sqlite for runs and the registry, a temporary
    folder for artifacts, no server. The experiment is created here with its artifact folder
    set, so nothing ever lands in the repo. Tests that register models use their own model
    name, so they don't see each other's versions or aliases."""
    root = tmp_path_factory.mktemp("mlflow")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{root}/mlflow.db")
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(root / "artifacts").as_uri())
    yield mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(previous)
