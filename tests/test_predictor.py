"""api.predictor.GrpcPredictor against a real local model server. The routes that use it
are covered in test_api.py; this file checks the proto -> ModelInfo/Scored mapping."""

from dataclasses import replace

import grpc
import pytest

from api.predictor import GrpcPredictor, ModelInfo
from services.model_server.main import WARMUP_ROW


@pytest.fixture
def predictor(model_server):
    p = GrpcPredictor(model_server.target, timeout_s=2, key=model_server.key)
    yield p
    p.close()


def test_info_maps_the_served_state(model_server, predictor, make_state):
    model_server.holder.set(make_state(version="4", threshold=0.25))
    assert predictor.ready()
    assert predictor.info() == ModelInfo(
        name="stroke-model", version="4", threshold=0.25, family="logreg", data_version="abc123"
    )


def test_unset_optional_fields_come_back_as_none(model_server, predictor, make_state):
    model_server.holder.set(replace(make_state(), family=None, data_version=None))
    info = predictor.info()
    assert (info.family, info.data_version) == (None, None)


def test_predict_takes_the_model_info_from_the_prediction(model_server, predictor, make_state):
    model_server.holder.set(make_state(version="9"))
    scored = predictor.predict(WARMUP_ROW, "r1")
    assert 0 <= scored.probability <= 1
    assert (scored.info.version, scored.info.family) == ("9", "logreg")


def test_not_ready_without_a_model_and_failed_calls_raise_rpc_errors(predictor):
    assert not predictor.ready()
    with pytest.raises(grpc.RpcError) as err:
        predictor.predict(WARMUP_ROW, "r1")
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE
