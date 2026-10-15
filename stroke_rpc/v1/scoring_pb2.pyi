from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Gender(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GENDER_UNSPECIFIED: _ClassVar[Gender]
    GENDER_MALE: _ClassVar[Gender]
    GENDER_FEMALE: _ClassVar[Gender]

class WorkType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORK_TYPE_UNSPECIFIED: _ClassVar[WorkType]
    WORK_TYPE_PRIVATE: _ClassVar[WorkType]
    WORK_TYPE_SELF_EMPLOYED: _ClassVar[WorkType]
    WORK_TYPE_GOVT_JOB: _ClassVar[WorkType]
    WORK_TYPE_CHILDREN: _ClassVar[WorkType]
    WORK_TYPE_NEVER_WORKED: _ClassVar[WorkType]

class ResidenceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESIDENCE_TYPE_UNSPECIFIED: _ClassVar[ResidenceType]
    RESIDENCE_TYPE_URBAN: _ClassVar[ResidenceType]
    RESIDENCE_TYPE_RURAL: _ClassVar[ResidenceType]

class SmokingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SMOKING_STATUS_UNSPECIFIED: _ClassVar[SmokingStatus]
    SMOKING_STATUS_UNKNOWN: _ClassVar[SmokingStatus]
    SMOKING_STATUS_FORMERLY_SMOKED: _ClassVar[SmokingStatus]
    SMOKING_STATUS_NEVER_SMOKED: _ClassVar[SmokingStatus]
    SMOKING_STATUS_SMOKES: _ClassVar[SmokingStatus]

class EverMarried(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVER_MARRIED_UNSPECIFIED: _ClassVar[EverMarried]
    EVER_MARRIED_YES: _ClassVar[EverMarried]
    EVER_MARRIED_NO: _ClassVar[EverMarried]
GENDER_UNSPECIFIED: Gender
GENDER_MALE: Gender
GENDER_FEMALE: Gender
WORK_TYPE_UNSPECIFIED: WorkType
WORK_TYPE_PRIVATE: WorkType
WORK_TYPE_SELF_EMPLOYED: WorkType
WORK_TYPE_GOVT_JOB: WorkType
WORK_TYPE_CHILDREN: WorkType
WORK_TYPE_NEVER_WORKED: WorkType
RESIDENCE_TYPE_UNSPECIFIED: ResidenceType
RESIDENCE_TYPE_URBAN: ResidenceType
RESIDENCE_TYPE_RURAL: ResidenceType
SMOKING_STATUS_UNSPECIFIED: SmokingStatus
SMOKING_STATUS_UNKNOWN: SmokingStatus
SMOKING_STATUS_FORMERLY_SMOKED: SmokingStatus
SMOKING_STATUS_NEVER_SMOKED: SmokingStatus
SMOKING_STATUS_SMOKES: SmokingStatus
EVER_MARRIED_UNSPECIFIED: EverMarried
EVER_MARRIED_YES: EverMarried
EVER_MARRIED_NO: EverMarried

class Patient(_message.Message):
    __slots__ = ("patient_id", "gender", "age", "hypertension", "heart_disease", "ever_married", "smoking_status", "work_type", "residence_type", "avg_glucose_level", "bmi")
    PATIENT_ID_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    AGE_FIELD_NUMBER: _ClassVar[int]
    HYPERTENSION_FIELD_NUMBER: _ClassVar[int]
    HEART_DISEASE_FIELD_NUMBER: _ClassVar[int]
    EVER_MARRIED_FIELD_NUMBER: _ClassVar[int]
    SMOKING_STATUS_FIELD_NUMBER: _ClassVar[int]
    WORK_TYPE_FIELD_NUMBER: _ClassVar[int]
    RESIDENCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    AVG_GLUCOSE_LEVEL_FIELD_NUMBER: _ClassVar[int]
    BMI_FIELD_NUMBER: _ClassVar[int]
    patient_id: str
    gender: Gender
    age: float
    hypertension: int
    heart_disease: int
    ever_married: EverMarried
    smoking_status: SmokingStatus
    work_type: WorkType
    residence_type: ResidenceType
    avg_glucose_level: float
    bmi: float
    def __init__(self, patient_id: _Optional[str] = ..., gender: _Optional[_Union[Gender, str]] = ..., age: _Optional[float] = ..., hypertension: _Optional[int] = ..., heart_disease: _Optional[int] = ..., ever_married: _Optional[_Union[EverMarried, str]] = ..., smoking_status: _Optional[_Union[SmokingStatus, str]] = ..., work_type: _Optional[_Union[WorkType, str]] = ..., residence_type: _Optional[_Union[ResidenceType, str]] = ..., avg_glucose_level: _Optional[float] = ..., bmi: _Optional[float] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("patient", "request_id")
    PATIENT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    patient: Patient
    request_id: str
    def __init__(self, patient: _Optional[_Union[Patient, _Mapping]] = ..., request_id: _Optional[str] = ...) -> None: ...

class PredictBatchRequest(_message.Message):
    __slots__ = ("requests",)
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[PredictRequest]
    def __init__(self, requests: _Optional[_Iterable[_Union[PredictRequest, _Mapping]]] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ("request_id", "probability", "high_risk", "threshold", "model_name", "model_version", "model_family")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    HIGH_RISK_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_FAMILY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    probability: float
    high_risk: bool
    threshold: float
    model_name: str
    model_version: str
    model_family: str
    def __init__(self, request_id: _Optional[str] = ..., probability: _Optional[float] = ..., high_risk: _Optional[bool] = ..., threshold: _Optional[float] = ..., model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., model_family: _Optional[str] = ...) -> None: ...

class ModelInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("name", "version", "threshold", "family", "data_version", "loaded_at_unix")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    FAMILY_FIELD_NUMBER: _ClassVar[int]
    DATA_VERSION_FIELD_NUMBER: _ClassVar[int]
    LOADED_AT_UNIX_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    threshold: float
    family: str
    data_version: str
    loaded_at_unix: int
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., threshold: _Optional[float] = ..., family: _Optional[str] = ..., data_version: _Optional[str] = ..., loaded_at_unix: _Optional[int] = ...) -> None: ...
