from dataclasses import dataclass
from typing import Any, Protocol

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

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
        model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
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
