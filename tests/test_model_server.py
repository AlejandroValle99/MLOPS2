"""services.model_server.main: how the server is put together (health, reflection, and the
reloader flipping health to SERVING), checked over a real local gRPC server."""

import grpc
import pandas as pd
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)

from services.model_server.main import SERVICE_NAME, WARMUP_ROW, mark_serving
from services.model_server.reloader import Reloader
from services.model_server.servicer import validated_record
from stroke_rpc.convert import patient_from_record

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING


def health_of(target: str, service: str = "") -> int:
    with grpc.insecure_channel(target) as channel:
        request = health_pb2.HealthCheckRequest(service=service)
        return health_pb2_grpc.HealthStub(channel).Check(request, timeout=2).status


def test_health_is_not_serving_until_the_first_model_loads(model_server, tiny_model):
    assert health_of(model_server.target) == NOT_SERVING
    assert health_of(model_server.target, SERVICE_NAME) == NOT_SERVING

    reloader = Reloader(
        model_server.holder,
        "stroke-model",
        lookup=lambda: ("1", {}),
        loader=lambda version: tiny_model,
        warmup_row=WARMUP_ROW,
        interval_s=10,
        on_loaded=mark_serving(model_server.health),
    )
    assert reloader.check_once() == "loaded"
    assert health_of(model_server.target) == SERVING
    assert health_of(model_server.target, SERVICE_NAME) == SERVING


def test_reflection_lists_the_scoring_and_health_services(model_server):
    with grpc.insecure_channel(model_server.target) as channel:
        services = set(ProtoReflectionDescriptorDatabase(channel).get_services())
    assert {"stroke_rpc.v1.Scoring", "grpc.health.v1.Health"} <= services


def test_the_warmup_row_is_a_valid_patient(tiny_model):
    """Every candidate model is scored on this row before a swap; it must itself be valid."""
    assert validated_record(patient_from_record(WARMUP_ROW)) == WARMUP_ROW
    probability = tiny_model.predict_proba(pd.DataFrame([WARMUP_ROW]))[0, 1]
    assert 0 <= probability <= 1
