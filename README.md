# Stroke Risk Service

Final project for **Operaciones de Aprendizaje Automático II** (MLOps II), CEIA · FIUBA.
Serves a stroke-risk prediction model over REST, GraphQL and gRPC, with streaming
inference, federated learning and a data lake added as the course progresses.

This builds on the author's own group project for MLOps I
(predicting stroke risk on the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)),
which supplies the Airflow + MLflow + MinIO + PostgreSQL base and the original
FastAPI service.

**Status:** the MLOps I base runs here with its known bugs fixed (among them, a DAG that
overwrote the raw dataset, now guarded by a checksum manifest), and the REST API is
complete (Mini-TP 1, below). The API scores through a gRPC model server that hot-reloads
the champion model when it changes. GraphQL, streaming, more rigorous training, federated
learning, the data lake and security are in progress.

## Architecture (the base; it grows as layers are added)

```
                        Docker Compose
┌────────────────────────────────────────────────────────────────┐
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
│   │   Airflow   │   │   MLflow    │   │      MinIO      │    │
│   │  :8080      │   │  :5001      │   │  :9000 / :9001  │    │
│   └──────┬──────┘   └──────┬──────┘   └────────┬────────┘    │
│          └────────┬────────┘                    │             │
│                   ▼                             │             │
│          ┌────────────────┐                     │             │
│          │   PostgreSQL   │◀────────────────────┘             │
│          │    :5432       │                                   │
│          └────────────────┘                                   │
│   ┌─────────────┐   gRPC   ┌──────────────┐                   │
│   │   FastAPI   │ ───────▶ │ Model server │ ◀── follows the   │
│   │   :8000     │          │   :50051     │     champion in   │
│   └─────────────┘          └──────────────┘     MLflow        │
│          ▲                                                    │
│          └── reads the dataset from MinIO                     │
└────────────────────────────────────────────────────────────────┘
```

| Dir | Role |
|---|---|
| `api/` | FastAPI REST service; a client of the model server |
| `proto/` | The gRPC contract (`stroke_rpc/v1/scoring.proto`) |
| `stroke_rpc/` | Generated gRPC code (`v1/`) and the row ↔ message conversion (`convert.py`) |
| `services/model_server/` | The gRPC model server: scoring, hot reload, health checks |
| `model/` | Feature pipeline and training |
| `dags/` | Airflow DAGs: data prep, model comparison, production retrain |
| `docker/` | Per-service Dockerfiles (Airflow, MLflow, Postgres init) |
| `stroke/` | Lifecycle modules added for MLOps II: the raw-data manifest and the frozen split today; evaluation, registry, drift and federated learning as they are built |
| `data/raw/` | The immutable source dataset + its integrity manifest |
| `scripts/` | Command-line clients and utilities (`mini_tp1_client.py`) |
| `tests/` | pytest suite, one file per module under test; shared fixtures in `conftest.py` |

## Setup

```bash
uv sync --group api --group ml --group train --group serve --group dev
```

## Run the full stack

```bash
cp .env.example .env   # fill in real values; .env is never committed
docker compose up -d
```

| Interface | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5001 |
| MinIO Console | http://localhost:9001 |

## REST API (Mini-TP 1)

The API serves the version of `stroke-model` that holds the `champion` alias in the
MLflow registry.

| Route | Purpose | API key |
|---|---|---|
| `POST /v1/predict` | One prediction | required |
| `GET /v1/model` | The served model: name, version, threshold, family, data version | required |
| `GET /health` | Liveness: 200 while the process is up | — |
| `GET /ready` | Readiness: 200 and the model version once a model is loaded, 503 before | — |

**Contract.** The request (`api/schemas.py`) holds the model's ten real features, with
range checks (for example `0 < age <= 120`). Unknown fields and out-of-range values are
rejected with a 422 that names the field. `bmi` may be omitted (the model imputes it), and
`patient_id` is optional and never used as a feature. The response carries the
probability, `high_risk` (probability at or above the model's threshold), the threshold,
and the model name and version.

**API keys.** The API accepts any of the comma-separated values in `API_KEYS`, sent in
the `X-API-KEY` header. Set `API_KEYS` in your env file.

**Client.** One valid call (200), one invalid call (422), and one call with a wrong key
(401), with each status and body printed:

```bash
API_KEY=<one of API_KEYS> uv run --group api python scripts/mini_tp1_client.py
```

## gRPC model server

The model lives in its own service, `model-server`, and the REST API calls it over gRPC.
A single process holds the model, so every client (the API today, the stream worker
later) scores with the same version, and only that process has to reload it.

| RPC | Shape | Use |
|---|---|---|
| `Predict` | unary: one patient in, one prediction out | the REST API's `/v1/predict` |
| `PredictBatch` | server streaming: many patients in, predictions streamed back in order | batch clients |
| `GetModelInfo` | unary | `/ready` and `/v1/model` |

**Contract.** `proto/stroke_rpc/v1/scoring.proto`. Categorical features are enums whose
`0` value is `UNSPECIFIED`, so a field the client forgot to set is rejected
(`INVALID_ARGUMENT`) rather than silently read as a real category. `bmi` is `optional`, so
a missing value stays missing and the model imputes it. The server applies the same range
checks as the REST contract. After editing the `.proto`, regenerate the code with
`./scripts/gen_proto.sh` and commit the result.

**Hot reload.** A background thread checks the registry's `champion` alias every
`RELOAD_INTERVAL_S` seconds (10 in compose). When the alias points at a new version, the
server loads that exact version and scores one sample patient with it. Only if both
succeed does it swap the new model in; otherwise it keeps serving the old one and does
not retry that version. Every prediction reports the `model_version` that produced it.

**Health.** The standard `grpc.health.v1` service reports `NOT_SERVING` until the first
model loads; Docker's healthcheck uses it, and the API waits for it. Server reflection is
on, so [`grpcurl`](https://github.com/fullstorydev/grpcurl) can explore the service:

```bash
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext -d '{"request_id": "demo", "patient": {"gender": "GENDER_MALE",
  "age": 67, "hypertension": 0, "heart_disease": 1, "ever_married": "EVER_MARRIED_YES",
  "work_type": "WORK_TYPE_PRIVATE", "residence_type": "RESIDENCE_TYPE_URBAN",
  "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "SMOKING_STATUS_FORMERLY_SMOKED"}}' \
  localhost:50051 stroke_rpc.v1.Scoring/Predict
```

**Errors seen through REST.** When a call to the model server fails, the API answers
422 (`INVALID_ARGUMENT`), 503 (`UNAVAILABLE`: server down or no model yet, or
`RESOURCE_EXHAUSTED`), 504 (`DEADLINE_EXCEEDED`: every call has an `RPC_TIMEOUT_S`
deadline, 2 s by default) or 502 (our own credentials towards the model server rejected).

Run it outside Docker, with the same MLflow/MinIO variables as the API below:

```bash
uv run --group ml --group serve python -m services.model_server.main
```

## Run the API locally (without Docker)

The rest of the stack must be up for the API to load the champion model. Stop the
compose `api` service first, because it also uses port 8000:

```bash
docker compose stop api
API_KEYS=dev-key \
MLFLOW_TRACKING_URI=http://localhost:5001 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER> AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD> \
uv run --group api --group ml uvicorn api.main:app --reload
```

## Tests

```bash
uv run --group api --group ml --group serve --group dev pytest
```

No running services are needed: the API tests train a small model in memory and swap it
in for the MLflow one.

## Lint, format, type check

```bash
uv run ruff check .
uv run ruff format .
uv run --group api --group ml --group serve mypy model stroke api services stroke_rpc
```

## Pre-commit hooks

Once per clone. Every commit then runs ruff (lint and format), mypy and the test suite,
using the same tool versions as `uv.lock`:

```bash
uv run pre-commit install
```
