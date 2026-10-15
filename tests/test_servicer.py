"""services.model_server.servicer: the Scoring RPCs, called over a real local gRPC server
(the `model_server` and `stub` fixtures in conftest.py) holding the `tiny_model` fixture."""

from dataclasses import replace

import grpc
import pandas as pd
import pytest

from api.schemas import StrokeInput
from model.preprocess import RAW_COLUMNS
from services.model_server.servicer import BOUNDS, CHUNK
from stroke_rpc.convert import request_from_record
from stroke_rpc.v1 import scoring_pb2 as pb

ROW = {
    "gender": "Female",
    "age": 61.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Self-employed",
    "Residence_type": "Rural",
    "avg_glucose_level": 202.21,
    "bmi": None,
    "smoking_status": "never smoked",
}


def rows_of(dataset, n: int) -> list[dict]:
    X, _ = dataset
    return [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in X[RAW_COLUMNS].head(n).to_dict("records")
    ]


def offline(model, rows: list[dict]) -> list[float]:
    return list(model.predict_proba(pd.DataFrame(rows))[:, 1])


def code_of(call) -> grpc.StatusCode:
    with pytest.raises(grpc.RpcError) as err:
        call()
    return err.value.code()


# --- Predict ----------------------------------------------------------------------------


def test_unary_matches_the_offline_model(model_server, stub, make_state, tiny_model, dataset):
    model_server.holder.set(make_state())
    for i, row in enumerate(rows_of(dataset, 20)):
        prediction = stub.Predict(request_from_record(row, f"r{i}"), timeout=2)
        assert prediction.probability == pytest.approx(offline(tiny_model, [row])[0])
        assert prediction.request_id == f"r{i}"


def test_prediction_carries_the_model_that_scored_it(model_server, stub, make_state):
    model_server.holder.set(make_state(version="7", threshold=0.0))
    prediction = stub.Predict(request_from_record(ROW, "r1"), timeout=2)
    assert prediction.model_name == "stroke-model"
    assert prediction.model_version == "7"
    assert prediction.model_family == "logreg"
    assert prediction.threshold == 0.0
    assert prediction.high_risk  # any probability is >= a threshold of 0


def test_a_missing_bmi_is_treated_as_missing_not_zero(model_server, stub, make_state, tiny_model):
    model_server.holder.set(make_state())
    request = request_from_record(ROW, "r1")
    assert not request.patient.HasField("bmi")
    prediction = stub.Predict(request, timeout=2)
    assert prediction.probability == pytest.approx(offline(tiny_model, [ROW])[0])


@pytest.mark.parametrize(
    "field", ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
)
def test_an_unset_enum_is_invalid_argument_naming_the_field(model_server, stub, make_state, field):
    model_server.holder.set(make_state())
    request = request_from_record(ROW, "r1")
    request.patient.ClearField(field)
    with pytest.raises(grpc.RpcError) as err:
        stub.Predict(request, timeout=2)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert field in err.value.details()


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", 0.0),  # also what an unset age reads as
        ("age", 130.0),
        ("avg_glucose_level", 0.0),
        ("avg_glucose_level", 600.0),
        ("bmi", 150.0),
        ("hypertension", 2),
        ("heart_disease", -1),
        ("patient_id", "x" * 65),
    ],
)
def test_out_of_range_values_are_invalid_argument(model_server, stub, make_state, field, value):
    model_server.holder.set(make_state())
    request = request_from_record(ROW, "r1")
    setattr(request.patient, field, value)
    with pytest.raises(grpc.RpcError) as err:
        stub.Predict(request, timeout=2)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert field in err.value.details()


def test_every_rpc_is_unavailable_before_a_model_is_loaded(stub):
    request = request_from_record(ROW, "r1")
    batch = pb.PredictBatchRequest(requests=[request])
    assert code_of(lambda: stub.Predict(request, timeout=2)) == grpc.StatusCode.UNAVAILABLE
    assert code_of(lambda: list(stub.PredictBatch(batch, timeout=2))) == (
        grpc.StatusCode.UNAVAILABLE
    )
    assert code_of(lambda: stub.GetModelInfo(pb.ModelInfoRequest(), timeout=2)) == (
        grpc.StatusCode.UNAVAILABLE
    )


# --- PredictBatch -----------------------------------------------------------------------


def test_batch_streams_every_result_in_order_across_chunks(
    model_server, stub, make_state, tiny_model, dataset
):
    model_server.holder.set(make_state())
    rows = rows_of(dataset, 2 * CHUNK + 50)  # three chunks, the last one partial
    batch = pb.PredictBatchRequest(
        requests=[request_from_record(row, f"r{i}") for i, row in enumerate(rows)]
    )
    results = list(stub.PredictBatch(batch, timeout=5))
    assert [p.request_id for p in results] == [f"r{i}" for i in range(len(rows))]
    assert [p.probability for p in results] == pytest.approx(offline(tiny_model, rows))


def test_a_promotion_between_two_batches_changes_the_version(model_server, stub, make_state):
    batch = pb.PredictBatchRequest(requests=[request_from_record(ROW, "r1")] * 3)
    model_server.holder.set(make_state(version="1"))
    first = list(stub.PredictBatch(batch, timeout=2))
    model_server.holder.set(make_state(version="2"))
    second = list(stub.PredictBatch(batch, timeout=2))
    assert {p.model_version for p in first} == {"1"}
    assert {p.model_version for p in second} == {"2"}


def test_one_bad_row_fails_the_whole_batch_and_names_it(model_server, stub, make_state):
    model_server.holder.set(make_state())
    bad = request_from_record(ROW, "bad-one")
    bad.patient.ClearField("gender")
    batch = pb.PredictBatchRequest(requests=[request_from_record(ROW, "ok"), bad])
    with pytest.raises(grpc.RpcError) as err:
        list(stub.PredictBatch(batch, timeout=2))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "requests[1]" in err.value.details()
    assert "bad-one" in err.value.details()
    assert "gender" in err.value.details()


def test_an_empty_batch_returns_no_results(model_server, stub, make_state):
    model_server.holder.set(make_state())
    assert list(stub.PredictBatch(pb.PredictBatchRequest(), timeout=2)) == []


# --- GetModelInfo -----------------------------------------------------------------------


def test_model_info_describes_the_served_model(model_server, stub, make_state):
    state = make_state(version="3", threshold=0.42)
    model_server.holder.set(state)
    info = stub.GetModelInfo(pb.ModelInfoRequest(), timeout=2)
    assert (info.name, info.version, info.threshold) == ("stroke-model", "3", 0.42)
    assert (info.family, info.data_version) == ("logreg", "abc123")
    assert info.loaded_at_unix == int(state.loaded_at)


def test_model_info_leaves_unknown_optional_fields_unset(model_server, stub, make_state):
    model_server.holder.set(replace(make_state(), family=None, data_version=None))
    info = stub.GetModelInfo(pb.ModelInfoRequest(), timeout=2)
    assert not info.HasField("family")
    assert not info.HasField("data_version")


def test_the_server_checks_the_same_bounds_as_the_rest_contract():
    """Both channels accept and refuse the same patients. The stream worker relies on it: it
    validates with the REST contract, so a row it passes must not be refused here."""
    for field, (low, high) in BOUNDS.items():
        limits = {type(m).__name__: m for m in StrokeInput.model_fields[field].metadata}
        assert (limits["Gt"].gt, limits["Le"].le) == (low, high), field
