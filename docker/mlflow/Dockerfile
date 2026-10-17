FROM python:3.12.15-slim-trixie

# Pinned to match the exact version resolved in uv.lock for every client (api, model-init,
# the Airflow image) — bug fixed here: this was previously an unpinned ">=2.15.0", so the
# server could drift to a different MLflow version than the clients talking to it.
RUN pip install --no-cache-dir \
    "mlflow==3.17.0" \
    psycopg2-binary \
    boto3

# Never root at run time; the server starts in its own home, the one folder it may write.
RUN useradd --create-home --uid 10001 mlflow
USER mlflow
WORKDIR /home/mlflow

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "*"
