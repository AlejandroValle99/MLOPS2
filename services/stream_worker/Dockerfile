FROM python:3.12.15-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
# stream: kafka-python, grpcio, protobuf, numpy, pandas and pydantic. No scikit-learn or
# mlflow: the worker never touches the model, it asks the model server.
RUN uv sync --frozen --no-dev --group stream
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Only what the worker imports: the REST contract (pydantic only), the drift math, the
# gRPC code, the worker itself, and the per-hospital drift baselines.
COPY api/__init__.py api/schemas.py ./api/
COPY stroke/__init__.py stroke/drift.py ./stroke/
COPY stroke_rpc/ ./stroke_rpc/
COPY services/__init__.py ./services/
COPY services/stream_worker/ ./services/stream_worker/
COPY data/drift/reference_stats.json ./data/drift/

# Never root at run time
RUN useradd --uid 10001 app
USER app

CMD ["python", "-m", "services.stream_worker.main"]
