"""The model card: what one version of the served model is, how good it is, and its limits.

Built from what the registry already records about a version (its tags, its run's
parameters and metrics, the gate's decision), plus the parts no number can say: what the
model is for, what it must not be used for, and what is known to be weak. Logged as
`model_card.json` in the version's run, next to `model/` but outside it, so the model's
files and their digest stay exactly as registered.
"""

import logging
from datetime import UTC, datetime

from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

CARD_NAME = "model_card.json"
OWNER = "ML Models and Something More Inc., MLOps team"

INTENDED_USE = (
    "Screening support for hospital clinicians and intake systems: a calibrated probability "
    "that an adult patient will have a stroke, and a high_risk flag (probability at or above "
    "the threshold) that marks who should be followed up first."
)
NOT_FOR = [
    "Diagnosing a stroke, or deciding a treatment without a clinician.",
    "Populations unlike the training data (one public dataset of 5,109 patients).",
    "Automated decisions that a patient cannot have reviewed by a person.",
]
LIMITATIONS = [
    "Small and imbalanced data: 249 strokes in total, 50 in the frozen test set, so every "
    "test metric is noisy and close models are hard to tell apart.",
    "Every gate decision reuses the same frozen test set, which slowly fits it over time.",
    "The four hospitals are simulated by a rule (residence type x age 60 and over); there is "
    "no real hospital column, so no per-hospital performance can be measured.",
    "The stream is a replay of the same dataset, with drift injected on purpose, and its "
    "events carry no outcome: monitoring covers inputs, not accuracy.",
    "The dataset's provenance and collection protocol are not documented.",
]
RISKS_AND_MITIGATIONS = [
    {
        "risk": "Unauthorized use of the model",
        "mitigation": "JWT with a role on REST; a key on every gRPC call; TLS to the model server",
    },
    {
        "risk": "Model extraction by scraping, and denial of service",
        "mitigation": "Per-caller rate limit and request size limit on REST; gRPC message size "
        "and concurrency limits",
    },
    {
        "risk": "Tampered model files (supply chain)",
        "mitigation": "artifact_sha256 checked before every load; skops loads only allowed "
        "types; promotion only through the statistical gate",
    },
    {
        "risk": "Tampered or poisoned training data",
        "mitigation": "Checksum manifest on the raw file, checked before every run",
    },
    {
        "risk": "Privacy leaks through logs and storage",
        "mitigation": "Pseudonymous patient ids; features never logged or stored; keyed "
        "input fingerprints in the audit log",
    },
    {
        "risk": "Adversarial evasion (inputs crafted to flip the flag)",
        "mitigation": "Not mitigated beyond input range checks; documented as a known gap",
    },
]

METRICS = ("f2", "recall", "precision", "roc_auc", "pr_auc", "brier", "ece", "alert_rate")


def _metrics(run: Run, prefix: str) -> dict[str, float]:
    found = run.data.metrics
    return {name: found[f"{prefix}_{name}"] for name in METRICS if f"{prefix}_{name}" in found}


def build_model_card(mv: ModelVersion, run: Run) -> dict:
    """mv = client.get_model_version(name, version); run = client.get_run(mv.run_id)."""
    tags = mv.tags
    hyperparameters = {
        key.removeprefix("hp_"): value
        for key, value in run.data.params.items()
        if key.startswith("hp_")
    }
    gate = {
        key.removeprefix("gate_"): value for key, value in tags.items() if key.startswith("gate_")
    }
    return {
        "model": f"{mv.name} v{mv.version}",
        "owner": OWNER,
        "registered_at": datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC).isoformat(),
        "card_written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "intended_use": INTENDED_USE,
        "not_for": NOT_FOR,
        "model_details": {
            "family": tags.get("model_family"),
            "threshold": float(tags["threshold"]) if "threshold" in tags else None,
            "calibration": "sigmoid, fitted on out-of-fold predictions",
            "threshold_choice": "maximizes F2 on out-of-fold calibrated probabilities",
            "hyperparameters": hyperparameters,
            "trained_by": run.data.tags.get("pipeline"),
            "artifact_sha256": tags.get("artifact_sha256"),
        },
        "data": {
            "data_version": tags.get("data_version"),
            "test_split": tags.get("test_split"),
            "features": "10 clinical and demographic features; patient ids are never features",
        },
        "metrics": {
            "out_of_fold": _metrics(run, "oof"),
            "test": _metrics(run, "test"),
            "note": "out-of-fold metrics chose the model; test metrics are for the record",
        },
        "gate": gate,
        "limitations": LIMITATIONS,
        "risks_and_mitigations": RISKS_AND_MITIGATIONS,
    }


def log_model_card(client: MlflowClient, name: str, version: str) -> dict:
    """Build the card of one version and log it to that version's run. Returns the card."""
    mv = client.get_model_version(name, version)
    if mv.run_id is None:
        raise ValueError(f"{name} v{version} has no run to log its card to")
    card = build_model_card(mv, client.get_run(mv.run_id))
    client.log_dict(mv.run_id, card, CARD_NAME)
    return card
