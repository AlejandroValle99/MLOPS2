"""FastAPI app — stroke prediction service."""

import io
import logging
import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from api.data import apply_mutations, fetch_dataset
from api.predictor import LocalPredictor, ModelInfo, Predictor
from api.schemas import StrokeInput, StrokePrediction

log = logging.getLogger(__name__)


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "stroke-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

_api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def require_api_key(key: Annotated[str | None, Depends(_api_key_header)]) -> str:
    allowed = [k for k in os.getenv("API_KEYS", "").split(",") if k]
    if key is None or not any(secrets.compare_digest(key, k) for k in allowed):
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    return key


ApiKeyDep = Annotated[str, Depends(require_api_key)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    predictor: Predictor
    try:
        predictor = LocalPredictor.from_registry(MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS)
    except Exception as exc:
        log.warning("no model loaded at startup: %s", exc)
        predictor = LocalPredictor()
    app.state.predictor = predictor
    yield


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


PredictorDep = Annotated[Predictor, Depends(get_predictor)]


app = FastAPI(title="Stroke Prediction API", version="0.1.0", lifespan=lifespan)

v1 = APIRouter(prefix="/v1")


# @app.get("/model-info")
# def model_info() -> dict:
#     try:
#         client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

#         model = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

#         run = client.get_run(model.run_id)

#         return {
#             "model_name": MODEL_NAME,
#             "model_alias": MODEL_ALIAS,
#             "model_version": model.version,
#             "model_run_id": model.run_id,
#             "test_f2": run.data.metrics.get("test_f2"),
#             "test_recall": run.data.metrics.get("test_recall"),
#             "test_precision": run.data.metrics.get("test_precision"),
#             "test_roc_auc": run.data.metrics.get("test_roc_auc"),
#         }

#     except Exception as exc:
#         raise HTTPException(
#             status_code=500, detail=f"Error al obtener información del modelo: {exc}"
#         ) from exc


@v1.post("/predict")
def predict(
    body: StrokeInput,
    predictor: PredictorDep,
    _key: ApiKeyDep,
    x_request_id: Annotated[str | None, Header()] = None,
) -> StrokePrediction:
    if not predictor.ready():
        raise HTTPException(status_code=503, detail="no model loaded")
    request_id = x_request_id or uuid4().hex

    scored = predictor.predict(body.to_model_row(), request_id)
    return StrokePrediction(
        request_id=request_id,
        probability=scored.probability,
        high_risk=scored.probability >= scored.info.threshold,
        threshold=scored.info.threshold,
        model_name=scored.info.name,
        model_version=scored.info.version,
        model_family=scored.info.family,
    )


@v1.get("/model")
def model(predictor: PredictorDep, _key: ApiKeyDep) -> ModelInfo:
    if not predictor.ready():
        raise HTTPException(status_code=503, detail="no model loaded")
    return predictor.info()


app.include_router(v1)


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(predictor: PredictorDep) -> dict:
    if not predictor.ready():
        raise HTTPException(status_code=503, detail="no model loaded")
    return {"status": "ready", "model_version": predictor.info().version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
