"""The REST API: the request contract, routes, API key, liveness vs readiness, and parity
with the offline model. No MLflow needed: the predictor is swapped for one holding
the `tiny_model` fixture (tests/conftest.py) through FastAPI's dependency_overrides.
"""

import time
from dataclasses import replace

import grpc
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_predictor
from api.predictor import GrpcPredictor, LocalPredictor, ModelInfo, Predictor
from scripts.mini_tp1_client import EXPECTED, VALID, run

KEY = {"X-API-KEY": "test-key"}
INFO = ModelInfo(
    name="stroke-model", version="7", threshold=0.5, family="logreg", data_version="abc123"
)


def serve(predictor: Predictor) -> None:
    """Make every route receive `predictor` instead of the one the lifespan would load."""
    app.dependency_overrides[get_predictor] = lambda: predictor


@pytest.fixture
def client(tiny_model, monkeypatch):
    monkeypatch.setenv("API_KEYS", "test-key,second-key")
    serve(LocalPredictor(tiny_model, INFO))
    yield TestClient(app)  # no `with`: the lifespan (and so MLflow) never runs
    app.dependency_overrides.clear()


def as_request_body(row: dict) -> dict:
    """A dataset row -> a /v1/predict body: snake_case residence_type, missing bmi as None."""
    body = {k: v for k, v in row.items() if k != "Residence_type"}
    body["residence_type"] = row["Residence_type"]
    if pd.isna(body["bmi"]):
        body["bmi"] = None  # NaN is not valid JSON
    return body


# --- the happy path ---------------------------------------------------------------------


def test_valid_call_returns_a_probability_and_the_model_version(client):
    r = client.post("/v1/predict", json=VALID, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["model_name"] == "stroke-model"
    assert body["model_version"] == "7"
    assert body["threshold"] == 0.5
    assert body["model_family"] == "logreg"


def test_request_id_is_echoed_from_the_header_or_generated(client):
    echoed = client.post("/v1/predict", json=VALID, headers={**KEY, "x-request-id": "req-42"})
    assert echoed.json()["request_id"] == "req-42"
    generated = client.post("/v1/predict", json=VALID, headers=KEY).json()["request_id"]
    assert generated  # a fresh id when the client sends none


@pytest.mark.parametrize("threshold, expected", [(0.0, True), (1.0, False)])
def test_high_risk_follows_the_model_threshold(client, tiny_model, threshold, expected):
    serve(LocalPredictor(tiny_model, replace(INFO, threshold=threshold)))
    body = client.post("/v1/predict", json=VALID, headers=KEY).json()
    assert body["high_risk"] is expected
    assert body["threshold"] == threshold


def test_bmi_may_be_omitted(client):
    without_bmi = {k: v for k, v in VALID.items() if k != "bmi"}
    r = client.post("/v1/predict", json=without_bmi, headers=KEY)
    assert r.status_code == 200


def test_patient_id_is_accepted_but_is_not_a_model_feature(client):
    r = client.post("/v1/predict", json={**VALID, "patient_id": "p-001"}, headers=KEY)
    assert r.status_code == 200


# --- the API key --------------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"X-API-KEY": "wrong"}], ids=["missing", "wrong"])
def test_missing_or_wrong_api_key_is_401(client, headers):
    assert client.post("/v1/predict", json=VALID, headers=headers).status_code == 401


def test_every_configured_api_key_is_accepted(client):
    r = client.post("/v1/predict", json=VALID, headers={"X-API-KEY": "second-key"})
    assert r.status_code == 200


# --- the contract rejects bad input with a 422 that names the field ----------------------


@pytest.mark.parametrize(
    "field, bad",
    [
        ("age", -1),
        ("age", 121),
        ("gender", "Other"),
        ("hypertension", 2),
        ("work_type", "Student"),
        ("avg_glucose_level", 600),
        ("bmi", 0),
        ("bmi", 101),
        ("smoking_status", "sometimes"),
        ("patient_id", "x" * 65),
    ],
)
def test_bad_input_is_422_and_names_the_field(client, field, bad):
    r = client.post("/v1/predict", json={**VALID, field: bad}, headers=KEY)
    assert r.status_code == 422
    assert field in r.json()["detail"][0]["loc"]


def test_unknown_field_is_422(client):
    # The old MLOps 1 field name is now an unknown field: extra="forbid" rejects it.
    r = client.post("/v1/predict", json={**VALID, "Residence_type": "Urban"}, headers=KEY)
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] == "extra_forbidden"


def test_missing_required_field_is_422(client):
    without_age = {k: v for k, v in VALID.items() if k != "age"}
    r = client.post("/v1/predict", json=without_age, headers=KEY)
    assert r.status_code == 422
    assert "age" in r.json()["detail"][0]["loc"]


# --- liveness vs readiness -----------------------------------------------------------------


def test_without_a_model_health_is_200_but_ready_and_predict_are_503(client):
    serve(LocalPredictor())
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post("/v1/predict", json=VALID, headers=KEY).status_code == 503


def test_ready_reports_the_loaded_version(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == "7"


def test_model_route_returns_the_model_info(client):
    r = client.get("/v1/model", headers=KEY)
    assert r.status_code == 200
    assert r.json() == {
        "name": "stroke-model",
        "version": "7",
        "threshold": 0.5,
        "family": "logreg",
        "data_version": "abc123",
    }


def test_model_route_is_503_without_a_model(client):
    serve(LocalPredictor())
    assert client.get("/v1/model", headers=KEY).status_code == 503


# --- golden parity: the API must score exactly like the model offline ---------------------


def test_api_probability_matches_offline_predict_proba(client, tiny_model, split):
    _, X_test, _, _ = split
    rows = X_test.head(50)
    expected = tiny_model.predict_proba(rows)[:, 1]
    for row, want in zip(rows.to_dict(orient="records"), expected, strict=True):
        r = client.post("/v1/predict", json=as_request_body(row), headers=KEY)
        assert r.status_code == 200
        assert r.json()["probability"] == pytest.approx(want, abs=1e-12)


# --- the Mini-TP 1 client itself -----------------------------------------------------------


def test_mini_tp1_client_shows_200_then_422_then_401(client):
    client.headers.update(KEY)  # the script's own main() sets this from API_KEY
    assert run(client) == EXPECTED


# --- through the gRPC model server (GrpcPredictor) -----------------------------------------


class SlowModel:
    """Answers correctly, but slower than the API's deadline."""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X):
        time.sleep(0.5)
        return self.model.predict_proba(X)


class FakeRpcError(grpc.RpcError):
    """What a failed stub call raises: an RpcError that also has code() and details()."""

    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return f"model server said {self._code.name}"


class FailingPredictor(LocalPredictor):
    def __init__(self, code: grpc.StatusCode) -> None:
        super().__init__(model=object(), info=INFO)
        self.code = code

    def predict(self, row: dict, request_id: str):
        raise FakeRpcError(self.code)


@pytest.fixture
def via_grpc(client, model_server, make_state):
    """The API wired to a real local model server, which starts out holding version 1."""
    model_server.holder.set(make_state(version="1"))
    predictor = GrpcPredictor(model_server.target, timeout_s=2)
    serve(predictor)
    yield model_server
    predictor.close()


def test_through_grpc_the_api_matches_offline_predict_proba(client, via_grpc, tiny_model, split):
    _, X_test, _, _ = split
    rows = X_test.head(20)
    expected = tiny_model.predict_proba(rows)[:, 1]
    for row, want in zip(rows.to_dict(orient="records"), expected, strict=True):
        r = client.post("/v1/predict", json=as_request_body(row), headers=KEY)
        assert r.status_code == 200
        assert r.json()["probability"] == pytest.approx(want, abs=1e-12)


def test_through_grpc_the_response_shows_the_version_that_scored_it(client, via_grpc, make_state):
    assert client.post("/v1/predict", json=VALID, headers=KEY).json()["model_version"] == "1"
    via_grpc.holder.set(make_state(version="2"))  # what a hot reload does
    assert client.post("/v1/predict", json=VALID, headers=KEY).json()["model_version"] == "2"


def test_through_grpc_ready_and_model_route_ask_the_model_server(client, via_grpc):
    assert client.get("/ready").json()["model_version"] == "1"
    assert client.get("/v1/model", headers=KEY).json()["data_version"] == "abc123"


def test_through_grpc_a_model_server_without_a_model_is_503(client, model_server):
    predictor = GrpcPredictor(model_server.target, timeout_s=2)
    serve(predictor)
    assert client.get("/ready").status_code == 503
    assert client.post("/v1/predict", json=VALID, headers=KEY).status_code == 503
    predictor.close()


def test_a_model_server_that_is_down_is_503(client):
    predictor = GrpcPredictor("localhost:1", timeout_s=1)  # nothing listens on port 1
    serve(predictor)
    assert client.get("/ready").status_code == 503
    assert client.post("/v1/predict", json=VALID, headers=KEY).status_code == 503
    predictor.close()


def test_a_slow_model_server_is_504(client, model_server, make_state, tiny_model):
    model_server.holder.set(make_state(model=SlowModel(tiny_model)))
    predictor = GrpcPredictor(model_server.target, timeout_s=0.2)
    serve(predictor)
    r = client.post("/v1/predict", json=VALID, headers=KEY)
    assert r.status_code == 504
    predictor.close()


@pytest.mark.parametrize(
    "code, http",
    [
        (grpc.StatusCode.INVALID_ARGUMENT, 422),
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, 503),
        (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
        (grpc.StatusCode.UNAUTHENTICATED, 502),
        (grpc.StatusCode.PERMISSION_DENIED, 502),
        (grpc.StatusCode.INTERNAL, 500),
    ],
)
def test_grpc_status_codes_map_to_http_statuses(client, code, http):
    serve(FailingPredictor(code))
    r = client.post("/v1/predict", json=VALID, headers=KEY)
    assert r.status_code == http
    assert r.json() == {"detail": f"model server said {code.name}"}
