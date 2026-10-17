FROM python:3.12.15-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
# ml: scikit-learn, xgboost and mlflow, so every model family the DAGs can promote unpickles
# here. serve: grpcio plus the health and reflection services. No FastAPI.
RUN uv sync --frozen --no-dev --group ml --group serve
ENV PATH="/app/.venv/bin:$PATH"
ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONDONTWRITEBYTECODE=1

# model/ holds the custom pipeline classes the pickled models reference (StrokeCleaner, ...)
COPY model/ ./model/
COPY stroke/ ./stroke/
COPY stroke_rpc/ ./stroke_rpc/
COPY services/__init__.py ./services/
COPY services/model_server/ ./services/model_server/

# Never root at run time: this process unpickles model files. The home holds MLflow's cache.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 50051

CMD ["python", "-m", "services.model_server.main"]
