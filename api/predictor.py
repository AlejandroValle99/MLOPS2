from dataclasses import dataclass
from typing import Any, Protocol

import grpc
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from stroke.registry import load_version
from stroke_rpc.channel import ClientTLS, open_channel
from stroke_rpc.convert import request_from_record
from stroke_rpc.v1 import scoring_pb2 as pb
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    threshold: float
    family: str | None = None
    data_version: str | None = None


@dataclass(frozen=True)
class Scored:
    probability: float
    info: ModelInfo


class Predictor(Protocol):
    def ready(self) -> bool: ...
    def info(self) -> ModelInfo: ...
    def predict(self, row: dict, request_id: str) -> Scored: ...


class LocalPredictor:
    def __init__(self, model: Any = None, info: ModelInfo | None = None) -> None:
        self._model = model
        self._info = info

    @classmethod
    def from_registry(cls, tracking_uri: str, name: str, alias: str) -> "LocalPredictor":
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri)
        mv = client.get_model_version_by_alias(name, alias)
        model = load_version(name, str(mv.version))  # checked against its artifact_sha256
        info = ModelInfo(
            name=name,
            version=str(mv.version),
            threshold=float(mv.tags.get("threshold", DEFAULT_THRESHOLD)),
            family=mv.tags.get("model_family"),
            data_version=mv.tags.get("data_version"),
        )
        return cls(model, info)

    def ready(self) -> bool:
        return self._model is not None

    def info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("no model loaded")
        return self._info

    def predict(self, row: dict, request_id: str) -> Scored:
        frame = pd.DataFrame([row])
        proba = self._model.predict_proba(frame)
        probability = float(proba[0, 1])
        return Scored(probability=probability, info=self.info())


class GrpcPredictor:
    """Same methods as LocalPredictor, but the model lives in the gRPC model server.

    A failed call raises grpc.RpcError; api/main.py maps its status code to an HTTP one.
    """

    def __init__(
        self,
        target: str,
        timeout_s: float,
        key: str | None = None,
        tls: ClientTLS | None = None,
    ) -> None:
        # One channel for the whole process: it keeps one connection open and multiplexes
        # every call over it, so opening a channel per request would only add latency.
        # `key` goes on every call; with `tls` the channel only talks to a server whose
        # certificate the configured CA signed.
        self.channel = open_channel(target, key=key, tls=tls)
        self.stub = pb_grpc.ScoringStub(self.channel)
        self.timeout_s = timeout_s  # every call gets a deadline, so none can hang forever

    def ready(self) -> bool:
        try:
            self.info()
        except grpc.RpcError as exc:
            if exc.code() in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
                raise  # our own key was refused: a 502, not "no model loaded"
            return False
        return True

    def info(self) -> ModelInfo:
        m = self.stub.GetModelInfo(pb.ModelInfoRequest(), timeout=self.timeout_s)
        return ModelInfo(
            name=m.name,
            version=m.version,
            threshold=m.threshold,
            family=m.family if m.HasField("family") else None,
            data_version=m.data_version if m.HasField("data_version") else None,
        )

    def predict(self, row: dict, request_id: str) -> Scored:
        p = self.stub.Predict(
            request_from_record(row, request_id),
            timeout=self.timeout_s,
            metadata=(("x-request-id", request_id),),
        )
        # The model info comes from the Prediction itself, not a separate info() call: after
        # a hot reload, that guarantees the version reported is the one that scored the row.
        info = ModelInfo(
            name=p.model_name,
            version=p.model_version,
            threshold=p.threshold,
            family=p.model_family if p.HasField("model_family") else None,
        )
        return Scored(probability=p.probability, info=info)

    def close(self) -> None:
        self.channel.close()
