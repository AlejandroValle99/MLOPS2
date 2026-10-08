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

**Status:** Phase 0 done — the MLOps I codebase has been moved in and its known bugs
fixed (see `docs/PLAN.md`, Phase 0 and Appendix C). Phases 1 onward (REST polish,
GraphQL, gRPC, streaming, training rigor, federated learning, data lake, security) are
in progress.

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

## Run the API locally (without Docker)

```bash
uv run --group api --group ml uvicorn api.main:app --reload
```

## Tests

```bash
uv run --group ml --group dev pytest
```

## Lint, format, type check

```bash
uv run ruff check .
uv run ruff format .
uv run mypy model stroke api
```

## Pre-commit hooks

```bash
uv run pre-commit install
```
