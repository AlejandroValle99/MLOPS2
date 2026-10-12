# Stroke Risk Service

Final project for **Operaciones de Aprendizaje Automático II** (MLOps II), CEIA · FIUBA.
Serves a stroke-risk prediction model over REST, GraphQL and gRPC, with streaming
inference, federated learning and a data lake added as the course progresses.

This builds on the author's own group project for MLOps I
(predicting stroke risk on the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)),
which supplies the Airflow + MLflow + MinIO + PostgreSQL base and the original
FastAPI service. See [`docs/PLAN.md`](docs/PLAN.md) for the full plan (goals, decisions,
phase-by-phase tasks) and [`docs/PRIMERS.md`](docs/PRIMERS.md) for primers on the
technologies new to this course (GraphQL, gRPC, streaming, federated learning, data
lakes, JWT/TLS).

**Status:** Phases 0 and 1 done. The MLOps I codebase was moved in and its known bugs
fixed (see `docs/PLAN.md`, Phase 0 and Appendix C), and the REST API was rebuilt for
Mini-TP 1 (below). Phases 2 onward (GraphQL, gRPC, streaming, training rigor, federated
learning, data lake, security) are in progress.

## Architecture (base, growing with each phase)

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
│   ┌─────────────┐                                             │
│   │   FastAPI   │  ◀── loads the champion model from MLflow   │
│   │   :8000     │  ◀── reads the dataset from MinIO           │
│   └─────────────┘                                             │
└────────────────────────────────────────────────────────────────┘
```

| Dir | Role |
|---|---|
| `api/` | FastAPI service: REST (and, from Phase 2, GraphQL) |
| `model/` | Feature pipeline and training |
| `dags/` | Airflow DAGs: data prep, model comparison, production retrain |
| `docker/` | Per-service Dockerfiles (Airflow, MLflow, Postgres init) |
| `stroke/` | New lifecycle modules added for MLOps II (data manifest/guard today; evaluation, registry, drift, federated learning as phases land) |
| `data/raw/` | The immutable source dataset + its integrity manifest |
| `docs/` | `PLAN.md` (the project plan) and `PRIMERS.md` (technology primers) |

## Setup

```bash
uv sync --group api --group ml --group train --group dev
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
uv run --group api --group ml --group dev pytest
```

No running services are needed: the API tests train a small model in memory and swap it
in for the MLflow one.

## Lint, format, type check

```bash
uv run ruff check .
uv run ruff format .
uv run mypy model stroke api
```

## Pre-commit hooks

Once per clone. Every commit then runs ruff (lint and format), mypy and the test suite,
using the same tool versions as `uv.lock`:

```bash
uv run pre-commit install
```
