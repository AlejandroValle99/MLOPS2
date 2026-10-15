FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
# lake: kafka-python, boto3 and pyarrow. Nothing else: the sink reads messages and writes
# files, it never scores or validates anything.
RUN uv sync --frozen --no-dev --group lake
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY services/__init__.py ./services/
COPY services/lake_sink/ ./services/lake_sink/

CMD ["python", "-m", "services.lake_sink.main"]
