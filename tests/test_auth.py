"""services.model_server.auth: every RPC shape the server uses needs the key; health doesn't.
Checked over the real local server from conftest.py (`model_server`), keyed with its key."""

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)

from services.model_server.auth import AuthInterceptor
from services.model_server.main import WARMUP_ROW
from stroke_rpc.channel import open_channel
from stroke_rpc.convert import request_from_record
from stroke_rpc.v1 import scoring_pb2 as pb
from stroke_rpc.v1 import scoring_pb2_grpc as pb_grpc

REQUEST = request_from_record(WARMUP_ROW, "r1")


def calls(stub: pb_grpc.ScoringStub) -> dict:
    """One call of each RPC, run to completion (a stream only fails once it is read)."""
    return {
        "Predict (unary)": lambda: stub.Predict(REQUEST, timeout=2),
        "PredictBatch (server streaming)": lambda: list(
            stub.PredictBatch(pb.PredictBatchRequest(requests=[REQUEST] * 3), timeout=2)
        ),
        "GetModelInfo (unary)": lambda: stub.GetModelInfo(pb.ModelInfoRequest(), timeout=2),
    }


@pytest.mark.parametrize("key", [None, "wrong-key"], ids=["no-key", "wrong-key"])
def test_every_rpc_without_the_right_key_is_unauthenticated(model_server, make_state, key):
    model_server.holder.set(make_state())
    with open_channel(model_server.target, key=key) as channel:
        for name, call in calls(pb_grpc.ScoringStub(channel)).items():
            with pytest.raises(grpc.RpcError) as err:
                call()
            assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED, name


def test_every_rpc_with_the_key_goes_through(model_server, make_state):
    model_server.holder.set(make_state())
    with open_channel(model_server.target, key=model_server.key) as channel:
        results = {name: call() for name, call in calls(pb_grpc.ScoringStub(channel)).items()}
    assert len(results["PredictBatch (server streaming)"]) == 3


def test_the_key_must_come_as_a_bearer_token(model_server, make_state):
    model_server.holder.set(make_state())
    with grpc.insecure_channel(model_server.target) as channel:
        stub = pb_grpc.ScoringStub(channel)
        with pytest.raises(grpc.RpcError) as err:
            stub.Predict(REQUEST, timeout=2, metadata=(("authorization", model_server.key),))
        assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED
        bearer = (("authorization", f"Bearer {model_server.key}"),)
        assert stub.Predict(REQUEST, timeout=2, metadata=bearer).model_version == "1"


def test_health_needs_no_key(model_server):
    with grpc.insecure_channel(model_server.target) as channel:
        response = health_pb2_grpc.HealthStub(channel).Check(
            health_pb2.HealthCheckRequest(), timeout=2
        )
    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_reflection_needs_the_key(model_server):
    with grpc.insecure_channel(model_server.target) as channel:
        with pytest.raises(grpc.RpcError) as err:
            ProtoReflectionDescriptorDatabase(channel).get_services()
    assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_a_server_without_keys_refuses_to_exist():
    with pytest.raises(ValueError, match="at least one key"):
        AuthInterceptor(["", ""])
