# MLOps 2

FastAPI service scaffold.

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn main:app --reload
```

## Lint & format

```bash
uv run ruff check .
uv run ruff format .
```

## Type check

```bash
uv run mypy main.py
```

## Pre-commit hooks

```bash
uv run pre-commit install
```

## Docker

```bash
docker build -t mlops-2 .
docker run -p 8000:8000 mlops-2
```
