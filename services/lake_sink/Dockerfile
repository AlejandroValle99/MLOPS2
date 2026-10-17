FROM python:3.12.15-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
# lake: kafka-python, boto3 and pyarrow. Nothing else: the sink reads messages and writes
# files, it never scores or validates anything.
RUN uv sync --frozen --no-dev --group lake
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY services/__init__.py ./services/
COPY services/lake_sink/ ./services/lake_sink/

# Never root at run time
RUN useradd --uid 10001 app
USER app

CMD ["python", "-m", "services.lake_sink.main"]
