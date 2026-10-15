FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

COPY api/ ./api/
COPY model/ ./model/
COPY stroke/ ./stroke/
# The generated gRPC code and convert.py: the API is a client of the model server.
COPY stroke_rpc/ ./stroke_rpc/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
