"""Mapping between dataset rows (records) and the protobuf messages in scoring.proto.

A record is one row with the dataset's column names (model.preprocess.RAW_COLUMNS), the
same shape StrokeInput.to_model_row() returns. Import only protobuf code here: the stream
worker imports this module, and its image has no scikit-learn.
"""

import math
from collections.abc import Mapping
from typing import Any

from stroke_rpc.v1 import scoring_pb2 as pb

GENDER = {pb.GENDER_MALE: "Male", pb.GENDER_FEMALE: "Female"}
EVER_MARRIED = {pb.EVER_MARRIED_YES: "Yes", pb.EVER_MARRIED_NO: "No"}
WORK_TYPE = {
    pb.WORK_TYPE_PRIVATE: "Private",
    pb.WORK_TYPE_SELF_EMPLOYED: "Self-employed",
    pb.WORK_TYPE_GOVT_JOB: "Govt_job",
    pb.WORK_TYPE_CHILDREN: "children",
    pb.WORK_TYPE_NEVER_WORKED: "Never_worked",
}
RESIDENCE_TYPE = {pb.RESIDENCE_TYPE_URBAN: "Urban", pb.RESIDENCE_TYPE_RURAL: "Rural"}
SMOKING_STATUS = {
    pb.SMOKING_STATUS_UNKNOWN: "Unknown",
    pb.SMOKING_STATUS_FORMERLY_SMOKED: "formerly smoked",
    pb.SMOKING_STATUS_NEVER_SMOKED: "never smoked",
    pb.SMOKING_STATUS_SMOKES: "smokes",
}


def _to_string(mapping: Mapping[Any, str], value: int, field: str) -> str:
    # UNSPECIFIED (0) is in no mapping, so an unset field lands here too
    if value not in mapping:
        raise ValueError(f"{field} is unset or has an unknown value ({value})")
    return mapping[value]


def _to_enum(mapping: Mapping[Any, str], value: str, field: str) -> Any:
    for enum_value, name in mapping.items():
        if name == value:
            return enum_value
    raise ValueError(f"{field} has an unknown value: {value!r}")


def record_from_patient(p: pb.Patient) -> dict:
    """proto Patient -> one row with the dataset's column names. ValueError on an unset enum."""
    return {
        "gender": _to_string(GENDER, p.gender, "gender"),
        "age": p.age,
        "hypertension": p.hypertension,
        "heart_disease": p.heart_disease,
        "ever_married": _to_string(EVER_MARRIED, p.ever_married, "ever_married"),
        "work_type": _to_string(WORK_TYPE, p.work_type, "work_type"),
        "Residence_type": _to_string(RESIDENCE_TYPE, p.residence_type, "residence_type"),
        "avg_glucose_level": p.avg_glucose_level,
        "bmi": p.bmi if p.HasField("bmi") else None,  # missing stays missing; the imputer fills it
        "smoking_status": _to_string(SMOKING_STATUS, p.smoking_status, "smoking_status"),
    }


def patient_from_record(record: dict) -> pb.Patient:
    """The reverse, for the clients (the API, the worker). Leave bmi unset when it is None."""
    patient = pb.Patient(
        patient_id=record.get("patient_id") or "",
        gender=_to_enum(GENDER, record["gender"], "gender"),
        age=record["age"],
        hypertension=record["hypertension"],
        heart_disease=record["heart_disease"],
        ever_married=_to_enum(EVER_MARRIED, record["ever_married"], "ever_married"),
        work_type=_to_enum(WORK_TYPE, record["work_type"], "work_type"),
        residence_type=_to_enum(RESIDENCE_TYPE, record["Residence_type"], "residence_type"),
        avg_glucose_level=record["avg_glucose_level"],
        smoking_status=_to_enum(SMOKING_STATUS, record["smoking_status"], "smoking_status"),
    )
    bmi = record.get("bmi")
    if bmi is not None and not math.isnan(bmi):  # dataset rows carry a missing bmi as NaN
        patient.bmi = bmi
    return patient


def request_from_record(record: dict, request_id: str) -> pb.PredictRequest:
    return pb.PredictRequest(patient=patient_from_record(record), request_id=request_id)
