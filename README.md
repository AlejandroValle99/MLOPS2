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
same model. Training calibrates every model and chooses its threshold, and a new model
replaces the champion only with statistical evidence that it is better. Every layer has
its security controls: tokens and roles on REST, TLS and keys to the model server, model
files checked before every load, an audit log and a model card per version (Security,
below).

## Architecture

```
clinician ──REST──▶ api :8000 ──gRPC Predict──▶ model-server :50051 ──▶ MLflow `champion`
                                                      ▲ gRPC PredictBatch
hospital systems ──▶ Redpanda ──▶ stream-worker ──────┘
                     (by hospital)      │
                                        ├──▶ predictions ──▶ each hospital reads its own
                                        ├──▶ drift-alerts
                                        └──▶ patient-events-dlq (rejected events)

Airflow :8080 ──▶ train, calibrate, gate ──▶ MLflow :5001 ──▶ model files in MinIO :9000
PostgreSQL :5432 holds Airflow's and MLflow's metadata. Everything runs in Docker Compose.
```

Only the API's port is open to the network; the others listen on loopback only. The API sits
on a Docker network of its own with the model server, so it can't reach MLflow or MinIO at
all. Calls to the model server go over TLS and carry the caller's key.

| Dir | Role |
|---|---|
| `api/` | FastAPI REST service; a client of the model server |
| `proto/` | The gRPC contract (`stroke_rpc/v1/scoring.proto`) |
| `stroke_rpc/` | Generated gRPC code (`v1/`) and the row ↔ message conversion (`convert.py`) |
| `services/model_server/` | The gRPC model server: scoring, hot reload, health checks |
| `services/stream_worker/` | The stream worker: validates intake events, scores them, publishes the scores, watches drift |
| `services/producer/` | Replays the dataset as intake events from the hospitals (run from the laptop) |
| `model/` | Feature pipeline and training (`train.py`: candidates, tuning, calibration, threshold) |
| `dags/` | Airflow DAGs: training with the promotion gate, a retrain on a fresh snapshot |
| `docker/` | Per-service Dockerfiles (Airflow, MLflow, Postgres init) |
| `stroke/` | Lifecycle modules added for MLOps II: the raw-data manifest and frozen split, the hospital rule, drift (PSI), evaluation (metrics, threshold, the gate's statistics), the registry (logging, the artifact digest, the gate, promotion, rollback), the audit log and the model card |
| `data/raw/` | The immutable source dataset + its integrity manifest |
| `data/drift/` | The per-hospital drift baselines the stream worker compares against (generated, committed) |
| `scripts/` | Command-line clients and utilities (`mini_tp1_client.py`, `hospital_inbox.py`, `rollback.py`, `record_artifact_digests.py`, `build_drift_baseline.py`, `make_certs.sh`, `gen_proto.sh`) |
| `tests/` | pytest suite, one file per module under test; shared fixtures in `conftest.py` |

## Setup

```bash
uv sync --group api --group ml --group train --group serve --group stream --group lake --group dev
```

## Run the full stack

```bash
cp .env.example .env   # fill in real values; .env is never committed
./scripts/make_certs.sh                     # once: the dev CA and TLS certificates, in certs/
docker compose up -d                        # the core stack
docker compose --profile streaming up -d    # plus the broker and the stream worker
```

Every value in `.env` must be replaced, and the empty ones (`JWT_SECRET`, `AUDIT_KEY`,
`MODEL_SERVER_KEY_API`, `MODEL_SERVER_KEY_WORKER`) filled:
`python -c "import secrets; print(secrets.token_urlsafe(32))"` makes one. The API refuses
to start with a secret under 32 bytes, and the model server without its keys.

| Interface | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5001 |
| MinIO Console | http://localhost:9001 |

## REST API (Mini-TP 1)

The API serves the version of `stroke-model` that holds the `champion` alias in the
MLflow registry.

| Route | Purpose | Role |
|---|---|---|
| `POST /v1/predict` | One prediction | `predict` or `admin` |
| `GET /v1/model` | The served model: name, version, threshold, family, data version | `admin` |
| `GET /health` | Liveness: 200 while the process is up | — |
| `GET /ready` | Readiness: 200 and the model version once a model is loaded, 503 before | — |

**Contract.** The request (`api/schemas.py`) holds the model's ten real features, with
range checks (for example `0 < age <= 120`). Unknown fields and out-of-range values are
rejected with a 422 that names the field. `bmi` may be omitted (the model imputes it), and
`patient_id` is optional and never used as a feature. The response carries the
probability, `high_risk` (probability at or above the model's threshold), the threshold,
and the model name and version.

**Tokens.** Every call but `/health` and `/ready` needs a JWT in the `Authorization:
Bearer` header (Swagger's **Authorize** button takes it). A token names its caller and one
role, `predict` (clinicians and hospital systems) or `admin` (operators), and expires (15
minutes by default). There is no login endpoint: whoever holds `JWT_SECRET` issues tokens,
here from inside the API's container, so the laptop never needs the secret:

```bash
docker compose exec api python -m api.security dr-house predict
docker compose exec api python -m api.security ops-1 admin --minutes 60
```

No token, a forged or expired one: 401. A valid token without the role: 403. Over
`RATE_LIMIT_PER_MINUTE` predictions in a minute (60 by default), per caller: 429 with
`Retry-After`. A body over 16 KB: 413.

**Client.** One valid call (200), one invalid call (422), and one call with a forged token
(401), with each status and body printed:

```bash
API_TOKEN=<a predict token> uv run --group api python scripts/mini_tp1_client.py
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
model loads; Docker's healthcheck uses it, and the API waits for it. Health is the one
service that needs no key.

**TLS and keys.** The server serves TLS only, with a certificate from the development CA
that `scripts/make_certs.sh` creates (valid for `model-server` and `localhost`), and every
other call must carry one of its keys as `authorization: Bearer <key>`: the API sends
`MODEL_SERVER_KEY_API`, the stream worker `MODEL_SERVER_KEY_WORKER`. Setting
`TLS_CLIENT_CA_FILE` turns on mutual TLS (clients then present `certs/client.crt`); compose
leaves it off. Messages over 1 MB are refused. Server reflection is on, behind the same
key, so [`grpcurl`](https://github.com/fullstorydev/grpcurl) can explore the service:

```bash
KEY=<MODEL_SERVER_KEY_API>
grpcurl -cacert certs/ca.crt -H "authorization: Bearer $KEY" localhost:50051 list
grpcurl -cacert certs/ca.crt -H "authorization: Bearer $KEY" \
  -d '{"request_id": "demo", "patient": {"gender": "GENDER_MALE",
  "age": 67, "hypertension": 0, "heart_disease": 1, "ever_married": "EVER_MARRIED_YES",
  "work_type": "WORK_TYPE_PRIVATE", "residence_type": "RESIDENCE_TYPE_URBAN",
  "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "SMOKING_STATUS_FORMERLY_SMOKED"}}' \
  localhost:50051 stroke_rpc.v1.Scoring/Predict
```

**Errors seen through REST.** When a call to the model server fails, the API answers
422 (`INVALID_ARGUMENT`), 503 (`UNAVAILABLE`: server down or no model yet, or
`RESOURCE_EXHAUSTED`), 504 (`DEADLINE_EXCEEDED`: every call has an `RPC_TIMEOUT_S`
deadline, 2 s by default) or 502 (our own credentials towards the model server rejected).

Run it outside Docker, with MLflow and MinIO reachable and at least one key (without
`TLS_CERT_FILE` and `TLS_KEY_FILE` it serves plaintext, and says so in its log):

```bash
MODEL_SERVER_KEYS=dev-key MLFLOW_TRACKING_URI=http://localhost:5001 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER> AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD> \
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

## Training and promotion

A model reaches patients only through one path: train every candidate, pick a finalist
on training data alone, then let a statistical gate decide whether it replaces the current
champion. Nothing moves the `champion` alias by hand except the documented rollback.

**One training run** (`model/train.py`), for each candidate family: `age_only` (logistic
regression on age alone, the floor every model has to beat), `logreg`, `random_forest`
and `xgboost`:

| Step | What happens | Why |
|---|---|---|
| Split | One stratified 80/20 split, fixed by seed, from the raw file after its checksum is verified | every model is judged on the same locked test rows |
| Tune | Optuna (TPE, median pruning) maximizes mean 5-fold CV PR-AUC on the training rows | PR-AUC needs no threshold, so it is steadier than F2 with ~200 positives |
| Calibrate | Sigmoid calibration on out-of-fold predictions | class weights inflate the scores on purpose; after calibration, 0.2 means about a 20% chance |
| Threshold | The cut that maximizes F2 on out-of-fold calibrated probabilities | F2 weighs recall four times precision, and out-of-fold rows were never seen by the model scoring them |
| Select | The finalist is the candidate with the best out-of-fold F2 | the test set picks nothing |
| Log | Hyperparameters (`hp_*`), `oof_*` and `test_*` metrics (F2, recall, precision, ROC-AUC, PR-AUC, Brier, ECE, alert rate), the model, and the per-hospital drift baseline | the test set is scored once per candidate, for the record |

**The gate** (`stroke/registry.py`). The finalist is registered as a new version of
`stroke-model`, carrying its `threshold`, `model_family`, `data_version`, `test_split` and
`artifact_sha256` tags (the digest of its files, below). Then:

- No champion yet: it is promoted.
- The champion was judged on another test set, or on none (a model from before the gate):
  the two can't be compared on equal terms, so the finalist is promoted and the reason says so.
- Otherwise both models score the test set, each at its own threshold, over 2,000 paired,
  stratified bootstrap resamples. The finalist is promoted only if its F2 is higher in at
  least 90% of them (`GATE_MIN_P_BETTER`). A tie is not a win.

Either way the version is tagged with the decision: `gate_decision`, `gate_reason`,
`gate_p_better`, `gate_mean_diff`, `gate_ci90`, both F2 scores, and the champion it was
compared with. A promotion keeps the old champion as `previous_champion`. Tags are set
before the alias moves, because the model server reads a version's threshold when it
loads it. Every model is logged set to predict on one thread: the model server scores one
patient (or a few hundred) per call, where a thread pool costs more than it saves. Each
gate decision also writes one audit line, and each version gets a model card
(`model_card.json` in its run): intended use, metrics, the gate's decision, limitations,
and risks with their mitigations.

**DAGs** (Airflow, http://localhost:8080; trigger them by hand):

| DAG | Does | Parameters |
|---|---|---|
| `etl_train_models_process_taskflow` | `verify_raw`, one `train_<family>` task per candidate in parallel, then `register_and_gate` | `n_trials` (30), `gate_min_p_better` (0.90) |
| `stroke_prediction_pipeline` | Retrains on a fresh snapshot (the same patients resampled, with noise), minus every frozen-test patient, then gates the finalist on the frozen test set | same |
| `0_stroke_full_pipeline` | Triggers the first one and waits for it | — |

`gate_min_p_better = 0` promotes the finalist whatever the comparison says. That is the
only way to force a promotion, and it is still recorded as a gate decision. On the first
start, `model-init` runs the same training and gate (10 trials per candidate) if there is
no champion yet.

**Rollback** puts `previous_champion` back, with its own threshold, within one reload
interval of the model server. Only aliases move. The runbook:

```bash
# 1. Roll back. Who ran it is tagged on the version rolled back from (rolled_back_by) and
#    written as an audit line, printed here.
MLFLOW_TRACKING_URI=http://localhost:5001 uv run --group ml python -m scripts.rollback
# 2. Within RELOAD_INTERVAL_S (10 s), the model server serves it:
docker compose logs --since 1m model-server | grep "now serving"
# 3. The API reports the same version and its own threshold (an admin token):
curl -s -H "Authorization: Bearer $ADMIN_TOKEN" localhost:8000/v1/model
```

Then check that new predictions (REST responses, a hospital's inbox) carry that
`model_version`. The script refuses a second rollback in a row: `previous_champion` is used
up. Moving the alias by hand in MLflow's UI does the same, but leaves no audit line.

**How to read the results.** On this dataset (5,109 patients, 249 strokes, 50 in the test
set) the candidates are close: age alone reaches a test ROC-AUC of 0.835, as high as the
full models (0.81 to 0.84), and the out-of-fold and test rankings often disagree. So most challengers are
rejected, and that is the gate working, not failing. Two limits to keep in mind: the
bootstrap measures the uncertainty of the test sample, not how differently another
training run (another seed, another snapshot) would have turned out; and every gate
decision reuses the same test set, so a long series of decisions slowly fits it. A real
deployment would refresh the test set over time.

## Security

Every threat below has a control and a test, or is marked out of scope with the reason.
The SAIF column names the element of Google's Secure AI Framework the control serves, as
the course maps them: strong foundations, detection and response, automated defenses,
harmonized controls, adapted controls, and risk in context.

| Threat | Control | SAIF | Test |
|---|---|---|---|
| Unauthorized use of the model (REST) | JWT (HS256, required `exp`, issuer and audience) with one role, `predict` or `admin`; the secret is at least 32 bytes | foundations | `test_api.py`: 401 for no, forged, unsigned, expired or foreign tokens; 403 for the wrong role |
| Unauthorized calls to the model server | a key per client on every call, checked by an interceptor on every RPC shape; health public, reflection keyed | foundations | `test_auth.py` |
| Eavesdropping between the API and the model server | TLS with a dev CA; mutual TLS available (`TLS_CLIENT_CA_FILE`) | foundations | `test_channel.py`: a plaintext client, one trusting another CA, and one without a client certificate on mTLS all fail with `UNAVAILABLE` |
| Model extraction by scraping, denial of service | per-caller rate limit (429), 16 KB body limit (413), 1 MB gRPC message limit, concurrent-RPC cap, a deadline on every call | automated defenses | `test_api.py` (429, 413), `test_security.py`, `test_model_server.py` |
| Injection and garbage input | the request contract (unknown fields and out-of-range values: 422), the same checks in the model server (`INVALID_ARGUMENT`), stream events validated, rejects dead-lettered | automated defenses | `test_api.py`, `test_servicer.py`, `test_stream_worker.py` |
| Tampered model files (supply chain) | `artifact_sha256`: every file of the model hashed right after logging, tagged on the version, checked before every load (model server, gate, local API); skops loads only allowed types; promotion only through the gate | harmonized controls | `test_registry.py`: a swapped `model.skops` and an edited `MLmodel` are refused while the old model keeps serving; a version without a digest is refused |
| Tampered training data | the raw file's checksum manifest, checked before every run; the retrain DAG checks the MinIO copy against it too | harmonized controls | `test_data.py` |
| Patient records through an open route | `/dataset` removed; the API holds no storage credentials and can't reach MinIO or MLflow (its own Docker network); internal ports on 127.0.0.1 | foundations | `test_api.py`: `/dataset` is 404 |
| Repudiation | the audit log: one JSON line per prediction, gate decision, rollback and refused call, with who, what and when | detection and response | `test_api.py`, `test_registry.py`, `test_audit.py` |
| Privacy leaks through logs | no feature ever logged; inputs appear only as an HMAC fingerprint (a plain hash of ten features could be reversed by trial); patient ids are the hospital's pseudonyms | foundations | `test_api.py`: no feature value in any log line, end to end |
| Secrets in git or images | every secret from `.env` (ignored by git and by Docker builds, like `certs/`); placeholders left empty | foundations | `test_api.py`: a short secret stops the API |
| A compromised container | every image built here runs as a non-root user, on pinned base images | foundations | — |
| An ungoverned model | a model card per version; the gate's evidence on every version's tags | risk in context | `test_model_card.py`, `test_train.py` |

**The audit log** is the `audit` logger in each service's output, one JSON object per
line: `docker compose logs api | grep ' audit '`. Actions: `predict`, `promote`, `reject`,
`rollback`, `record-digest`, `auth-denied`, `forbidden`, `rate-limited`. A prediction line
holds the caller, the request id, the patient's pseudonym, the input's fingerprint, the
model version and `high_risk`; never a feature, and not the probability, which the input
and the version reproduce. Stream predictions are recorded by the `predictions` topic and
the lake, under the same rules.

**Out of scope, and why:**

- *MLflow, MinIO and Redpanda accounts:* MLflow and Redpanda run without authentication,
  and every service uses MinIO's root user. They are reachable only from inside their
  Docker networks and from this machine. A deployment would put MLflow behind its auth app
  or SSO, give each service its own MinIO user and bucket policy, and turn on SASL and
  ACLs in the broker, which is also what one topic per hospital needs.
- *A human approval before promotion:* the gate promotes on statistical evidence alone.
  An `approved_by` check only means something once MLflow itself enforces who may move an
  alias; until then anyone who reaches it could move the alias around the check.
- *Signed stream events* (an HMAC per hospital): with no broker authentication, a
  hospital's identity rests on the key and the contract checks; signatures would come with
  per-hospital credentials.
- *Adversarial evasion:* inputs are range-checked, but no adversarial training was done.
  The model card lists it as a known gap.
- *Token issuing:* tokens come from a command line with the shared secret. A deployment
  would use an identity provider, and asymmetric keys, so the API holds only a public key.

**A registry from before the digest.** Versions registered earlier have no
`artifact_sha256`, and the model server refuses them. Record theirs once, before starting
the new model server, then start (or restart) it:

```bash
docker compose run --rm model-init python -m scripts.record_artifact_digests
```

That vouches for the files as they are at that moment, so do it once, never to make a
refused version load: a refusal means its files changed after the digest was recorded.

## Not in this project, and why

The course covers more layers than this service uses; each one here has a job.

- **GraphQL** was considered for reading model metadata, and left out: there is one
  consumer, a handful of model versions, and MLflow's UI already shows them.
- **Federated learning** was left out: the service trains on data it holds, so no
  hospital needs to train without sharing records, and here a federation could only have
  been simulated in one process.

## Run the API locally (without Docker)

The rest of the stack must be up for the API to load the champion model. Stop the
compose `api` service first, because it also uses port 8000. Without `MODEL_SERVER_TARGET`
the API loads the champion itself (after checking its digest), so it needs MLflow and
MinIO:

```bash
docker compose stop api
JWT_SECRET=<32+ bytes> AUDIT_KEY=<32+ bytes> \
MLFLOW_TRACKING_URI=http://localhost:5001 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=<MINIO_ROOT_USER> AWS_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD> \
uv run --group api --group ml uvicorn api.main:app --reload
JWT_SECRET=<the same> uv run --group api --group ml python -m api.security me predict
```

## Tests

```bash
uv run --group api --group ml --group train --group serve --group stream --group lake --group dev pytest
```

No running services are needed: the tests train small models in memory, run a real gRPC
model server on a local port (with TLS from a temporary CA where TLS is what is tested;
those tests need `openssl`), use a throwaway MLflow (sqlite and a temporary folder) for
the registry and the gate, and stand in for the broker with fakes. The XGBoost tests skip
on macOS when `libomp` is missing (`brew install libomp`).

## Lint, format, type check

```bash
uv run ruff check .
uv run ruff format .
uv run --group api --group ml --group train --group serve --group stream --group lake mypy model stroke api services stroke_rpc
```

## Pre-commit hooks

Once per clone. Every commit then runs ruff (lint and format), mypy and the test suite,
using the same tool versions as `uv.lock`:

```bash
uv run pre-commit install
```
