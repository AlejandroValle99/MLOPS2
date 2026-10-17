"""The REST API: the request contract, routes, tokens and roles, rate and size limits, the
audit line, liveness vs readiness, and parity with the offline model. No MLflow needed:
the predictor is swapped for one holding the `tiny_model` fixture (tests/conftest.py)
through FastAPI's dependency_overrides.
"""

import json
import logging
import time
from dataclasses import replace

import grpc
import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_predictor
from api.predictor import GrpcPredictor, LocalPredictor, ModelInfo, Predictor
from api.security import RateLimiter, issue_token
from scripts.mini_tp1_client import EXPECTED, VALID, run

SECRET = "test-jwt-secret-that-is-long-enough-for-hs256"
AUDIT = "test-audit-key-that-is-long-enough-for-hmac"


def bearer(role: str = "predict", subject: str = "clinician-1", **kwargs) -> dict:
    return {"Authorization": f"Bearer {issue_token(subject, role, secret=SECRET, **kwargs)}"}


KEY = bearer("predict")  # a clinician
ADMIN = bearer("admin", "ops-1")
INFO = ModelInfo(
    name="stroke-model", version="7", threshold=0.5, family="logreg", data_version="abc123"
)


def serve(predictor: Predictor) -> None:
    """Make every route receive `predictor` instead of the one the lifespan would load."""
    app.dependency_overrides[get_predictor] = lambda: predictor


@pytest.fixture
def client(tiny_model, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("AUDIT_KEY", AUDIT)
    # A fresh limiter per test, high enough that only the rate-limit tests ever reach it
    monkeypatch.setattr(app.state, "rate_limiter", RateLimiter(per_minute=10_000))
    serve(LocalPredictor(tiny_model, INFO))
    yield TestClient(app)  # no `with`: the lifespan (and so MLflow) never runs
    app.dependency_overrides.clear()


def audit_lines(caplog) -> list[dict]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "audit"]


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


# --- tokens and roles: 401 is "who are you?", 403 is "not you" --------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": f"Bearer {issue_token('x', 'predict', secret='another-secret' * 4)}"},
    ],
    ids=["missing", "garbage", "wrong-scheme", "wrong-signature"],
)
def test_a_missing_or_invalid_token_is_401(client, headers):
    r = client.post("/v1/predict", json=VALID, headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_an_expired_token_is_401(client):
    r = client.post("/v1/predict", json=VALID, headers=bearer(minutes=-1))
    assert r.status_code == 401
    assert r.json()["detail"] == "token expired"


def test_a_token_for_another_audience_is_401(client):
    forged = jwt.encode(
        {"sub": "x", "role": "admin", "iat": time.time(), "exp": time.time() + 60}, SECRET
    )  # signed with the right secret, but no issuer or audience: not one of ours
    r = client.get("/v1/model", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


def test_an_unsigned_token_is_401(client):
    claims = {"sub": "x", "role": "admin", "iss": "stroke-risk", "aud": "stroke-api"}
    claims |= {"iat": time.time(), "exp": time.time() + 60}
    unsigned = jwt.encode(claims, None, algorithm="none")  # "alg": "none", no signature
    r = client.get("/v1/model", headers={"Authorization": f"Bearer {unsigned}"})
    assert r.status_code == 401


def test_both_roles_may_predict(client):
    assert client.post("/v1/predict", json=VALID, headers=KEY).status_code == 200
    assert client.post("/v1/predict", json=VALID, headers=ADMIN).status_code == 200


def test_the_predict_role_cannot_read_the_model_details(client):
    r = client.get("/v1/model", headers=KEY)
    assert r.status_code == 403
    assert client.get("/v1/model", headers=ADMIN).status_code == 200


def test_the_api_refuses_a_short_secret(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "change_me_to_32_bytes_or_more")  # 29 bytes
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        with TestClient(app):  # runs the lifespan, which checks the secrets first
            pass


# --- rate and size limits --------------------------------------------------------------------


def test_the_call_over_the_rate_limit_is_429(client, monkeypatch):
    monkeypatch.setattr(app.state, "rate_limiter", RateLimiter(per_minute=3))
    statuses = [client.post("/v1/predict", json=VALID, headers=KEY).status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    r = client.post("/v1/predict", json=VALID, headers=KEY)
    assert 0 < int(r.headers["retry-after"]) <= 60
    # The limit is per caller: another clinician still gets through
    other = bearer(subject="clinician-2")
    assert client.post("/v1/predict", json=VALID, headers=other).status_code == 200


def test_an_oversized_body_is_413_before_it_is_read(client):
    huge = {**VALID, "patient_id": "x" * 20_000}
    assert client.post("/v1/predict", json=huge, headers=KEY).status_code == 413


def test_the_dataset_route_is_gone(client):
    assert client.get("/dataset").status_code == 404


# --- the audit log -------------------------------------------------------------------------------


def test_every_prediction_leaves_one_audit_line_without_features(client, caplog):
    caplog.set_level(logging.INFO, logger="audit")
    body = {**VALID, "patient_id": "p-001"}
    r = client.post("/v1/predict", json=body, headers={**KEY, "x-request-id": "req-7"})
    (line,) = audit_lines(caplog)
    assert line["actor"] == "clinician-1"
    assert line["action"] == "predict"
    assert (line["request_id"], line["patient_id"]) == ("req-7", "p-001")
    assert line["model_version"] == "7"
    assert line["high_risk"] == r.json()["high_risk"]
    assert len(line["input"]) == 32  # a keyed fingerprint of the features...
    for feature, value in VALID.items():  # ...and none of the features themselves
        assert feature not in line
        if isinstance(value, float):
            assert str(value) not in json.dumps(line)


def test_refused_calls_are_audited(client, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="audit")
    client.post("/v1/predict", json=VALID)
    client.get("/v1/model", headers=KEY)
    monkeypatch.setattr(app.state, "rate_limiter", RateLimiter(per_minute=0))
    client.post("/v1/predict", json=VALID, headers=KEY)
    actions = [(line["actor"], line["action"]) for line in audit_lines(caplog)]
    assert actions == [
        ("anonymous", "auth-denied"),
        ("clinician-1", "forbidden"),
        ("clinician-1", "rate-limited"),
    ]


def test_no_log_line_holds_a_feature_value(client, via_grpc, caplog):
    """Through the whole path (REST, gRPC, the model server), at every log level."""
    caplog.set_level(logging.DEBUG)
    distinctive = {**VALID, "avg_glucose_level": 213.37, "bmi": 31.41, "age": 71.5}
    assert client.post("/v1/predict", json=distinctive, headers=KEY).status_code == 200
    too_old = {**distinctive, "age": 130}
    assert client.post("/v1/predict", json=too_old, headers=KEY).status_code == 422
    assert caplog.records  # something was logged
    for value in ("213.37", "31.41", "71.5"):
        assert value not in caplog.text


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
    r = client.get("/v1/model", headers=ADMIN)
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
    assert client.get("/v1/model", headers=ADMIN).status_code == 503


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
    client.headers.update(KEY)  # the script's own main() sets this from API_TOKEN
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
    predictor = GrpcPredictor(model_server.target, timeout_s=2, key=model_server.key)
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
    assert client.get("/v1/model", headers=ADMIN).json()["data_version"] == "abc123"


def test_a_wrong_upstream_key_is_502_not_401(client, model_server, make_state):
    """The model server refusing the API's own key is our fault, not the caller's."""
    model_server.holder.set(make_state())
    predictor = GrpcPredictor(model_server.target, timeout_s=2, key="not-the-key")
    serve(predictor)
    r = client.post("/v1/predict", json=VALID, headers=KEY)
    assert r.status_code == 502
    predictor.close()


def test_through_grpc_a_model_server_without_a_model_is_503(client, model_server):
    predictor = GrpcPredictor(model_server.target, timeout_s=2, key=model_server.key)
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
    predictor = GrpcPredictor(model_server.target, timeout_s=0.2, key=model_server.key)
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
