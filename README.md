# Stroke Risk Service

Final project for **Operaciones de Aprendizaje Automático II** (MLOps II), CEIA · FIUBA.
A stroke-risk prediction service for hospitals. Clinicians score one patient at a time
over REST; hospital systems publish patient-intake events to a stream and read each
patient's score back. A gRPC model server holds the one model both channels use, and
every hospital's incoming population is watched for drift.

This builds on the author's own group project for MLOps I
(predicting stroke risk on the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)),
which supplies the Airflow + MLflow + MinIO + PostgreSQL base and the original
FastAPI service.

**Status:** the MLOps I base runs here with its known bugs fixed (among them, a DAG that
overwrote the raw dataset, now guarded by a checksum manifest), and the REST API is
complete (Mini-TP 1, below). The API scores through a gRPC model server that hot-reloads
the champion model when it changes, and hospital systems can stream intake events to the
same model. More rigorous training, the data lake and security are in progress.

## Architecture

```
clinician ──REST──▶ api :8000 ──gRPC Predict──▶ model-server :50051 ──▶ MLflow `champion`
                                                      ▲ gRPC PredictBatch
hospital systems ──▶ Redpanda ──▶ stream-worker ──────┘
                     (by hospital)      │
                                        ├──▶ predictions ──▶ each hospital reads its own
                                        ├──▶ drift-alerts
                                        └──▶ patient-events-dlq (rejected events)

Airflow :8080 ──▶ train, compare, promote ──▶ MLflow :5001 ──▶ model files in MinIO :9000
PostgreSQL :5432 holds Airflow's and MLflow's metadata. Everything runs in Docker Compose.
```

| Dir | Role |
|---|---|
| `api/` | FastAPI REST service; a client of the model server |
| `proto/` | The gRPC contract (`stroke_rpc/v1/scoring.proto`) |
| `stroke_rpc/` | Generated gRPC code (`v1/`) and the row ↔ message conversion (`convert.py`) |
| `services/model_server/` | The gRPC model server: scoring, hot reload, health checks |
| `services/stream_worker/` | The stream worker: validates intake events, scores them, publishes the scores, watches drift |
| `services/producer/` | Replays the dataset as intake events from the hospitals (run from the laptop) |
| `model/` | Feature pipeline and training |
| `dags/` | Airflow DAGs: data prep, model comparison, production retrain |
| `docker/` | Per-service Dockerfiles (Airflow, MLflow, Postgres init) |
| `stroke/` | Lifecycle modules added for MLOps II: the raw-data manifest, the frozen split, the hospital rule and drift (PSI); evaluation and the registry as they are built |
| `data/raw/` | The immutable source dataset + its integrity manifest |
| `data/drift/` | The per-hospital drift baselines the stream worker compares against (generated, committed) |
| `scripts/` | Command-line clients and utilities (`mini_tp1_client.py`, `hospital_inbox.py`, `build_drift_baseline.py`, `gen_proto.sh`) |
| `tests/` | pytest suite, one file per module under test; shared fixtures in `conftest.py` |

## Setup

```bash
uv sync --group api --group ml --group train --group serve --group stream --group dev
```

## Run the full stack

```bash
cp .env.example .env   # fill in real values; .env is never committed
docker compose up -d                        # the core stack
docker compose --profile streaming up -d    # plus the broker and the stream worker
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
A single process holds the model, so every client (the API and the stream worker) scores
with the same version, and only that process has to reload it.

| RPC | Shape | Use |
|---|---|---|
| `Predict` | unary: one patient in, one prediction out | the REST API's `/v1/predict` |
| `PredictBatch` | server streaming: many patients in, predictions streamed back in order | the stream worker, one call per poll |
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

## Streaming: intake events from hospital systems

REST serves a person who asks about one patient and waits. Hospitals' own systems work
differently: they already emit an event whenever a patient is registered. Those systems
integrate through a stream instead. They publish each intake event to Redpanda (a
Kafka-compatible broker), the stream worker scores it, and the hospital reads the score
back from the `predictions` topic. The hospital never waits on the service, and if the
worker is down, events wait in the broker for up to an hour (the topic's retention, kept
short because the events hold patient data) and are scored when it is back.

This isn't about speed: stroke-risk features change slowly, and a nightly file scored in
batch would serve these hospitals just as well. The stream is here because it matches how
hospital systems already integrate, and because it decouples them from the service's
uptime. The dataset has no hospital column, so four simulated hospitals stand in for the
customers: residence type × age 60 and over (`stroke.data.assign_hospital`).

| Topic | Key | Holds |
|---|---|---|
| `patient-events` | hospital | `{hospital_id, event_id, event_time, patient}`; `patient` is exactly a `/v1/predict` request body. One-hour retention |
| `predictions` | hospital | the score: event id, pseudonymous patient id, probability, `high_risk`, threshold, model name and version. No patient features |
| `drift-alerts` | hospital | a sustained population shift: hospital, feature, PSI, window, model versions |
| `patient-events-dlq` | hospital | events that failed the contract, as received, with the reason. One-hour retention |

**Validation and the dead-letter topic.** Every event is checked against the same contract
as the REST API before it is scored. A failing event (bad JSON, an out-of-range value, an
unknown hospital, a message key that isn't the event's hospital) goes to
`patient-events-dlq` instead, with the reason and the event's first 4 KB, and the rest of
the batch is scored. Without this, one malformed event would block the stream: the model
server refuses a whole batch when one row is invalid, so the same batch would fail on
every retry.

**Delivery: at least once.** The worker scores each poll's events in one `PredictBatch`
call, publishes the results, waits until the broker has acknowledged every one of them,
and only then commits its offsets. If the broker refuses any message, the worker stops
before committing and, once restarted, reads those events again. A crash can therefore
repeat a few predictions, never lose one. If the model server fails for any reason (down,
slow, a model that crashes, rejected credentials), the worker rewinds to the batch's first
event and retries it after a pause. Only a batch the server refuses as invalid, which the
worker's own check should already have caught, is dead-lettered rather than retried.

**Drift.** Each hospital's scored events fill a window of `WINDOW_EVENTS` events (300 by
default). When it closes, the worker compares it with that hospital's own baseline
(`data/drift/reference_stats.json`, built from the frozen training split) using the
Population Stability Index on age, glucose, BMI, work type and smoking status. It logs
one JSON line per window (throughput, latency p50/p95, positive rate, PSI). PSI above 0.25
in two windows in a row publishes an alert. Patient features exist only in transit and in
the worker's memory: nothing that is stored holds them.

**Limits, stated plainly.** The events are a replay of the dataset, and drift is injected
on purpose. All hospitals share one `predictions` topic keyed by hospital; a real
deployment would give each its own topic, readable only with its own credentials. Patient
ids are pseudonyms (keyed with a secret drawn per run), but each event still carries the
patient's features, so this is pseudonymisation, not anonymisation. The events carry no
outcome either, as at a real registration, so the stream monitors inputs, not accuracy.

Run it (the core stack must be up):

```bash
docker compose --profile streaming up -d
# a hospital's inbox: its high-risk patients, as they are scored
uv run --group stream python scripts/hospital_inbox.py rural-60plus
# in another terminal: replay the dataset as intake events, 100 per second
uv run --group ml --group stream python -m services.producer.main --rate 100
# inject drift: glucose x1.2 at one hospital, from the 2,000th event overall (every
# hospital's events count, so that is 20 s in at the default 100 events/s)
uv run --group ml --group stream python -m services.producer.main \
  --drift-hospital rural-60plus --drift-from 2000 --factor 1.2
# send every 500th event invalid, to watch the dead-letter topic
uv run --group ml --group stream python -m services.producer.main --bad-every 500
# watch the worker's windows, and the alerts and rejected events
docker compose logs -f stream-worker
docker compose exec redpanda rpk topic consume drift-alerts patient-events-dlq
```

After a change to the split, the cleaning or the drift code, regenerate the baselines
(a test fails until you do):

```bash
uv run --group ml python -m scripts.build_drift_baseline
```

## Not in this project, and why

The course covers more layers than this service uses; each one here has a job.

- **GraphQL** was considered for reading model metadata, and left out: there is one
  consumer, a handful of model versions, and MLflow's UI already shows them.
- **Federated learning** was left out: the service trains on data it holds, so no
  hospital needs to train without sharing records, and here a federation could only have
  been simulated in one process.

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
uv run --group api --group ml --group serve --group stream --group dev pytest
```

No running services are needed: the tests train a small model in memory, run a real gRPC
model server on a local port, and stand in for the broker with fakes.

## Lint, format, type check

```bash
uv run ruff check .
uv run ruff format .
uv run --group api --group ml --group serve --group stream mypy model stroke api services stroke_rpc
```

## Pre-commit hooks

Once per clone. Every commit then runs ruff (lint and format), mypy and the test suite,
using the same tool versions as `uv.lock`:

```bash
uv run pre-commit install
```
