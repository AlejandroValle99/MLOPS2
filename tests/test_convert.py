"""stroke_rpc.convert: dataset rows <-> the protobuf messages in scoring.proto."""

import math

import pytest

from model.preprocess import RAW_COLUMNS
from stroke_rpc import convert
from stroke_rpc.v1 import scoring_pb2 as pb

ROW = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Self-employed",
    "Residence_type": "Rural",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


def test_every_dataset_row_survives_a_round_trip(dataset):
    X, _ = dataset
    for row in X[RAW_COLUMNS].to_dict("records"):
        back = convert.record_from_patient(convert.patient_from_record(row))
        if math.isnan(row["bmi"]):  # NaN != NaN, so compare a missing bmi on its own
            assert back.pop("bmi") is None
            row.pop("bmi")
        assert back == row


def test_record_has_the_raw_columns_in_order():
    record = convert.record_from_patient(convert.patient_from_record(ROW))
    assert list(record) == RAW_COLUMNS


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_a_missing_bmi_stays_unset_instead_of_becoming_zero(missing):
    patient = convert.patient_from_record({**ROW, "bmi": missing})
    assert not patient.HasField("bmi")
    assert convert.record_from_patient(patient)["bmi"] is None


@pytest.mark.parametrize(
    "field", ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
)
def test_an_unset_enum_is_rejected_and_names_the_field(field):
    patient = convert.patient_from_record(ROW)
    patient.ClearField(field)
    with pytest.raises(ValueError, match=field):
        convert.record_from_patient(patient)


def test_an_unknown_category_string_is_rejected():
    with pytest.raises(ValueError, match="work_type"):
        convert.patient_from_record({**ROW, "work_type": "Freelance"})


@pytest.mark.parametrize(
    ("mapping", "enum"),
    [
        (convert.GENDER, pb.Gender),
        (convert.EVER_MARRIED, pb.EverMarried),
        (convert.WORK_TYPE, pb.WorkType),
        (convert.RESIDENCE_TYPE, pb.ResidenceType),
        (convert.SMOKING_STATUS, pb.SmokingStatus),
    ],
)
def test_every_enum_value_except_unspecified_has_a_dataset_string(mapping, enum):
    """Fails when a value is added to the .proto but not to its dict here."""
    assert set(mapping) == {v for v in enum.values() if v != 0}


def test_request_from_record_carries_the_request_id_and_patient_id():
    request = convert.request_from_record({**ROW, "patient_id": "p-7"}, "req-1")
    assert request.request_id == "req-1"
    assert request.patient.patient_id == "p-7"
    assert convert.patient_from_record(ROW).patient_id == ""  # the API's row has none
