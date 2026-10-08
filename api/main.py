"""FastAPI app — stroke prediction service."""

import io
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlflow.tracking import MlflowClient

from api.data import apply_mutations, fetch_dataset
from api.schemas import StrokeInput, StrokePrediction

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "stroke-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        _model = mlflow.sklearn.load_model(model_uri)
        print(f"Modelo cargado desde {model_uri}")
    except Exception as exc:
        print(f"Advertencia: no se pudo cargar el modelo: {exc}")
    yield
    _model = None


app = FastAPI(title="Stroke Prediction API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    if _model is None:
        raise HTTPException(
            status_code=503, detail={"status": "unavailable", "model_loaded": False}
        )
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info() -> dict:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        model = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        run = client.get_run(model.run_id)

        return {
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": model.version,
            "model_run_id": model.run_id,
            "test_f2": run.data.metrics.get("test_f2"),
            "test_recall": run.data.metrics.get("test_recall"),
            "test_precision": run.data.metrics.get("test_precision"),
            "test_roc_auc": run.data.metrics.get("test_roc_auc"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener información del modelo: {exc}"
        ) from exc


@app.post("/predict", response_model=StrokePrediction)
def predict(data: StrokeInput) -> StrokePrediction:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Entrená primero el pipeline de Airflow.",
        )
    df = pd.DataFrame([data.model_dump()])
    prob = float(_model.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    return StrokePrediction(stroke_prediction=pred, stroke_probability=prob)


@app.get("/dataset")
def dataset() -> StreamingResponse:
    try:
        df = fetch_dataset()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not fetch dataset from MinIO: {exc}"
        ) from exc
    df = apply_mutations(df)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
