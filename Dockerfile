FROM python:3.12.15-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Shared by the api and model-init services (docker-compose.yml): both need the
# api group (FastAPI) and the ml group (scikit-learn/xgboost/mlflow); model-init
# additionally needs the train group (Optuna) for the comparison DAG's bootstrap run.
# Bug fixed here (was missing): xgboost was absent from this image, so an XGBoost
# champion promoted by the comparison DAG could never load in the API.
RUN uv sync --frozen --no-dev --group api --group ml --group train
ENV PATH="/app/.venv/bin:$PATH"
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONDONTWRITEBYTECODE=1

COPY api/ ./api/
COPY model/ ./model/
COPY stroke/ ./stroke/
# The generated gRPC code and convert.py: the API is a client of the model server.
COPY stroke_rpc/ ./stroke_rpc/
# The operator scripts that run in model-init's container, next to MLflow and MinIO
# (python -m scripts.record_artifact_digests)
COPY scripts/ ./scripts/

# Never root at run time. The home holds MLflow's and matplotlib's caches.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000

# uvicorn itself, not `uv run`: uv would try to sync the root-owned environment first
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
