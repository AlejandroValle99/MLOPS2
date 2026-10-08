from typing import Literal

from pydantic import BaseModel, Field


class StrokeInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., examples=["Male"])
    age: float = Field(..., gt=0, examples=[67.0])
    hypertension: int = Field(..., ge=0, le=1, examples=[0])
    heart_disease: int = Field(..., ge=0, le=1, examples=[1])
    ever_married: Literal["Yes", "No"] = Field(..., examples=["Yes"])
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"] = Field(
        ..., examples=["Private"]
    )
    Residence_type: Literal["Urban", "Rural"] = Field(..., examples=["Urban"])
    avg_glucose_level: float = Field(..., gt=0, examples=[228.69])
    bmi: float | None = Field(None, gt=0, examples=[36.6])
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"] = Field(
        ..., examples=["formerly smoked"]
    )


class StrokePrediction(BaseModel):
    stroke_prediction: int
    stroke_probability: float
