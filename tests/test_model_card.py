"""stroke.model_card: built from what the registry records about a version, and logged in
its run, outside the model's own files."""

import uuid

import mlflow
from mlflow.tracking import MlflowClient

from stroke.model_card import CARD_NAME, log_model_card
from stroke.registry import digest_of, log_candidate, register, use_experiment


def test_the_card_holds_the_versions_facts_and_sits_beside_its_model(
    mlflow_store, tiny_model, split
):
    X_train = split[0]
    client = MlflowClient(mlflow_store)
    name = f"stroke-card-{uuid.uuid4().hex[:8]}"
    use_experiment()
    logged = log_candidate(
        "logreg",
        tiny_model,
        params={"C": 0.5},
        metrics={"oof_f2": 0.41, "test_f2": 0.38, "test_ece": 0.02},
        tags={"pipeline": "test"},
        reference_stats={},
        input_example=X_train.head(1),
        pip_requirements=[],
    )
    version = register(
        name,
        {
            "model_uri": logged.model_uri,
            "artifact_sha256": logged.artifact_sha256,
            "threshold": 0.12,
            "model_family": "logreg",
            "data_version": "abc123",
            "test_split": "abc123-seed42-test0.2",
        },
    )
    client.set_model_version_tag(name, version, "gate_decision", "promoted")

    card = log_model_card(client, name, version)

    assert card["model"] == f"{name} v{version}"
    details = card["model_details"]
    assert (details["family"], details["threshold"]) == ("logreg", 0.12)
    assert details["hyperparameters"] == {"C": "0.5"}
    assert details["trained_by"] == "test"
    assert details["artifact_sha256"] == logged.artifact_sha256
    assert card["data"]["test_split"] == "abc123-seed42-test0.2"
    assert card["metrics"]["out_of_fold"] == {"f2": 0.41}
    assert card["metrics"]["test"] == {"f2": 0.38, "ece": 0.02}
    assert card["gate"] == {"decision": "promoted"}
    assert card["intended_use"] and card["limitations"] and card["risks_and_mitigations"]

    run_id = client.get_model_version(name, version).run_id
    assert mlflow.artifacts.load_dict(f"runs:/{run_id}/{CARD_NAME}") == card
    # Next to the model, not inside it: the model's files, and so its digest, are unchanged
    assert digest_of(logged.model_uri) == logged.artifact_sha256
