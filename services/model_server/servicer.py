"""The Scoring service: Predict (unary), PredictBatch (server streaming), GetModelInfo."""

from collections.abc import Iterator

import grpc
import pandas as pd

from services.model_server.holder import ModelHolder, ModelState
from stroke_rpc.convert import record_from_patient
from stroke_rpc.v1 import scoring_pb2 as pb
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

CHUNK = 256  # rows per predict_proba call inside one PredictBatch

# (low, high]: the same bounds as the REST contract in api/schemas.py
BOUNDS = {"age": (0, 120), "avg_glucose_level": (0, 500), "bmi": (0, 100)}


def validated_record(patient: pb.Patient) -> dict:
    """Patient -> dataset row, checked against the REST contract's limits. proto3 only
    checks types, so every bound has to be enforced here. ValueError if one is broken."""
    record = record_from_patient(patient)
    if len(patient.patient_id) > 64:
        raise ValueError("patient_id is longer than 64 characters")
    for field, (low, high) in BOUNDS.items():
        value = record[field]
        if value is not None and not low < value <= high:  # bmi may be None (missing)
            raise ValueError(f"{field} must be in ({low}, {high}], got {value}")
    for field in ("hypertension", "heart_disease"):
        if record[field] not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1, got {record[field]}")
    return record


def _prediction(request_id: str, probability: float, state: ModelState) -> pb.Prediction:
    prediction = pb.Prediction(
        request_id=request_id,
        probability=probability,
        high_risk=probability >= state.threshold,
        threshold=state.threshold,
        model_name=state.name,
        model_version=state.version,
    )
    if state.family is not None:
        prediction.model_family = state.family
    return prediction


class ScoringServicer(pb_grpc.ScoringServicer):
    def __init__(self, holder: ModelHolder) -> None:
        self.holder = holder

    def _state_or_abort(self, context: grpc.ServicerContext) -> ModelState:
        state = self.holder.get()
        if state is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "no model loaded yet")
            raise AssertionError("unreachable")  # abort() already raised; this line is for mypy
        return state

    def Predict(self, request: pb.PredictRequest, context: grpc.ServicerContext) -> pb.Prediction:
        state = self._state_or_abort(context)  # read the holder ONCE per call
        try:
            row = validated_record(request.patient)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise AssertionError("unreachable") from exc
        probability = float(state.model.predict_proba(pd.DataFrame([row]))[0, 1])
        return _prediction(request.request_id, probability, state)

    def PredictBatch(
        self, request: pb.PredictBatchRequest, context: grpc.ServicerContext
    ) -> Iterator[pb.Prediction]:
        # One state for the whole stream: a hot swap mid-batch must not mix model versions.
        state = self._state_or_abort(context)
        # Validate every row before streaming any result, so a bad row fails the whole call
        # up front instead of after half the predictions were already sent.
        rows = []
        for i, item in enumerate(request.requests):
            try:
                rows.append(validated_record(item.patient))
            except ValueError as exc:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"requests[{i}] (request_id={item.request_id!r}): {exc}",
                )
        for start in range(0, len(rows), CHUNK):
            if not context.is_active():
                return  # the client cancelled or its deadline passed: stop scoring
            chunk = request.requests[start : start + CHUNK]
            probabilities = state.model.predict_proba(pd.DataFrame(rows[start : start + CHUNK]))
            for item, probability in zip(chunk, probabilities[:, 1], strict=True):
                yield _prediction(item.request_id, float(probability), state)

    def GetModelInfo(
        self, request: pb.ModelInfoRequest, context: grpc.ServicerContext
    ) -> pb.ModelInfo:
        state = self._state_or_abort(context)
        info = pb.ModelInfo(
            name=state.name,
            version=state.version,
            threshold=state.threshold,
            loaded_at_unix=int(state.loaded_at),
        )
        if state.family is not None:
            info.family = state.family
        if state.data_version is not None:
            info.data_version = state.data_version
        return info
