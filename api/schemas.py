from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrokeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str | None = Field(None, max_length=64)
    gender: Literal["Male", "Female"] = Field(..., examples=["Male"])
    age: float = Field(..., le=120, gt=0, examples=[67.0])
    hypertension: int = Field(..., ge=0, le=1, examples=[0])
    heart_disease: int = Field(..., ge=0, le=1, examples=[1])
    ever_married: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"] = Field(
        ..., examples=["Private"]
    )
    residence_type: Literal["Urban", "Rural"] = Field(..., examples=["Urban"])
    avg_glucose_level: float = Field(..., le=500, gt=0, examples=[228.69])
    bmi: float | None = Field(None, le=100, gt=0, examples=[36.6])
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"] = Field(
        ..., examples=["formerly smoked"]
    )

    def to_model_row(self) -> dict:
        row = self.model_dump(exclude={"patient_id"})
        row["Residence_type"] = row.pop("residence_type")
        return row


# response type
class StrokePrediction(BaseModel):
    request_id: str
    probability: float
    high_risk: bool
    threshold: float
    model_name: str
    model_version: str
