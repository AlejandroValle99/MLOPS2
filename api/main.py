"""FastAPI app — stroke prediction service.

The API is the service's public edge, and it holds no data: no dataset, no model files,
no storage credentials. It checks who is calling (api/security.py), validates the request
(api/schemas.py), and asks the model server for the score over gRPC.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

import grpc
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from api.predictor import GrpcPredictor, LocalPredictor, ModelInfo, Predictor
from api.schemas import StrokeInput, StrokePrediction
from api.security import (
    BodySizeLimit,
    RateLimiter,
    audit_key,
    jwt_secret,
    rate_limited,
    require_role,
)
from stroke.audit import audit_event, input_fingerprint
from stroke_rpc.channel import ClientTLS

log = logging.getLogger(__name__)


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "stroke-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# What a REST caller sees when a call to the model server fails. Anything else is a 500.
GRPC_TO_HTTP = {
    grpc.StatusCode.INVALID_ARGUMENT: 422,  # bad input reached the model server
    grpc.StatusCode.UNAVAILABLE: 503,  # model server down, or no model loaded yet
    grpc.StatusCode.RESOURCE_EXHAUSTED: 503,  # model server shedding load
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,  # slower than our deadline
    # Our own credentials towards the model server are wrong: not the caller's fault,
    # so never a 401 (that would tell the caller to fix a key they don't control).
    grpc.StatusCode.UNAUTHENTICATED: 502,
    grpc.StatusCode.PERMISSION_DENIED: 502,
}

# A /v1/predict body is a few hundred bytes; nothing the API accepts comes near this.
MAX_BODY_BYTES = 16 * 1024

# Who called, already checked: the token's claims (sub, role, ...)
PredictCaller = Annotated[dict, Depends(rate_limited)]
Admin = Annotated[dict, Depends(require_role("admin"))]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # The audit lines are INFO records on the `audit` logger; uvicorn configures only its own.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Refuse to start with a missing or short secret, rather than fail on the first call
    jwt_secret()
    audit_key()
    predictor: Predictor
    target = os.getenv("MODEL_SERVER_TARGET")
    if target:
        # The model lives in the gRPC model server, which also hot-reloads it. Every call
        # carries the API's own key, over TLS when TLS_CA_FILE is set.
        predictor = GrpcPredictor(
            target,
            float(os.getenv("RPC_TIMEOUT_S", "2")),
            key=os.getenv("UPSTREAM_API_KEY"),
            tls=ClientTLS.from_env(),
        )
    else:
        # No model server configured: load the champion into this process, once.
        try:
            predictor = LocalPredictor.from_registry(MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS)
        except Exception as exc:
            log.warning("no model loaded at startup: %s", exc)
            predictor = LocalPredictor()
    app.state.predictor = predictor
    yield
    if isinstance(predictor, GrpcPredictor):
        predictor.close()


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


PredictorDep = Annotated[Predictor, Depends(get_predictor)]


app = FastAPI(title="Stroke Prediction API", version="0.1.0", lifespan=lifespan)
app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)
app.state.rate_limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))


@app.exception_handler(grpc.RpcError)
async def grpc_error(request: Request, exc: grpc.RpcError) -> JSONResponse:
    # An RpcError raised by a call is also a grpc.Call, which is where code() and details() live
    status = GRPC_TO_HTTP.get(exc.code(), 500)
    # Only the code in the log: details() can quote feature values ("age ... got 130.0"),
    # and patient features must never end up in logs. The caller still gets the details.
    log.warning("model server call failed: %s -> HTTP %d", exc.code(), status)
    return JSONResponse(status_code=status, content={"detail": exc.details()})


v1 = APIRouter(prefix="/v1")


@v1.post("/predict")
def predict(
    body: StrokeInput,
    predictor: PredictorDep,
    caller: PredictCaller,
    x_request_id: Annotated[str | None, Header(max_length=64)] = None,
) -> StrokePrediction:
    if not predictor.ready():
        raise HTTPException(status_code=503, detail="no model loaded")
    request_id = x_request_id or uuid4().hex

    row = body.to_model_row()
    scored = predictor.predict(row, request_id)
    high_risk = scored.probability >= scored.info.threshold
    audit_event(
        caller["sub"],
        "predict",
        request_id=request_id,
        patient_id=body.patient_id,  # the hospital's pseudonym, or None
        input=input_fingerprint(row, audit_key()),  # never the features themselves
        model_version=scored.info.version,
        high_risk=high_risk,
    )
    return StrokePrediction(
        request_id=request_id,
        probability=scored.probability,
        high_risk=high_risk,
        threshold=scored.info.threshold,
        model_name=scored.info.name,
        model_version=scored.info.version,
        model_family=scored.info.family,
    )


@v1.get("/model")
def model(predictor: PredictorDep, _admin: Admin) -> ModelInfo:
    """The served model's details (family, data version, threshold): for operators only."""
    if not predictor.ready():
        raise HTTPException(status_code=503, detail="no model loaded")
    return predictor.info()


app.include_router(v1)


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
