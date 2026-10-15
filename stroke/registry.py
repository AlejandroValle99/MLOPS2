"""The model registry's conventions: how a candidate is logged, how a finalist becomes a
version, and the only code that moves the `champion` alias (the gate, and the rollback).

Every version of the served model goes through `run_gate`. A version carries the tags the
model server reads when it loads it (threshold, model_family, data_version), the tag that
says which test set it was judged on (test_split), and the gate's decision (gate_*). Tags
are always set before the alias moves: the model server reads a version's tags once, when
it loads that version, so a tag added afterwards would not reach it until it restarts.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from skops.io import dumps, get_untrusted_types

from stroke.config import (
    CHAMPION,
    EXPERIMENT_NAME,
    GATE_BOOTSTRAPS,
    GATE_MIN_P_BETTER,
    PREVIOUS_CHAMPION,
)
from stroke.evaluation import paired_bootstrap

log = logging.getLogger(__name__)

# The only code the serving images have. A logged model may only reference types from here.
TRUSTED_PREFIXES = ("sklearn.", "xgboost.", "model.preprocess.")

# Tags set on a version when it is registered, from the finalist's summary.
VERSION_TAGS = ("threshold", "model_family", "data_version", "test_split")

# What MLflow answers when an alias simply isn't set (or the model doesn't exist yet).
NO_ALIAS_CODES = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def trusted_types_for(model: Any) -> list[str]:
    """The types skops has to be told to trust when it loads this model.

    skops refuses to load any type it doesn't trust by default, so MLflow saves this list
    with the model. Each type must come from scikit-learn, xgboost or model.preprocess:
    those are all the serving images can import. Anything else (a class defined in a DAG
    file or a notebook) would only fail later, inside the model server, so it fails here.
    """
    untrusted = get_untrusted_types(data=dumps(model))
    unexpected = [t for t in untrusted if not t.startswith(TRUSTED_PREFIXES)]
    if unexpected:
        raise ValueError(f"the model references types the serving images can't load: {unexpected}")
    return untrusted


def use_experiment(name: str = EXPERIMENT_NAME) -> str:
    """Make `name` the active experiment, creating it if needed. Returns its id."""
    try:
        return mlflow.set_experiment(name).experiment_id
    except MlflowException:
        # Parallel DAG tasks can race to create it; the loser finds it on the second try.
        return mlflow.set_experiment(name).experiment_id


def log_candidate(
    name: str,
    model: Any,
    params: dict,
    metrics: dict[str, float],
    tags: dict[str, str],
    reference_stats: dict,
    input_example: pd.DataFrame,
    pip_requirements: list[str] | None = None,  # None: MLflow works them out
) -> tuple[str, str]:
    """One MLflow run for one trained candidate, with the model artifact (not registered:
    only the finalist becomes a version). Returns (run_id, model_uri)."""
    with mlflow.start_run(run_name=name, tags={"model_family": name, **tags}) as run:
        mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(reference_stats, "reference_stats.json")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=trusted_types_for(model),
            input_example=input_example,
            pip_requirements=pip_requirements,
        )
    return run.info.run_id, info.model_uri


def register(name: str, finalist: dict) -> str:
    """A new version of `name` from the finalist's logged model, with its serving tags set
    in the same call that creates it. Returns the version, as a string."""
    tags = {key: str(finalist[key]) for key in VERSION_TAGS}
    mv = mlflow.register_model(finalist["model_uri"], name, tags=tags)
    return str(mv.version)  # MLflow returns versions as ints; compare them as strings


def get_alias(client: MlflowClient, name: str, alias: str = CHAMPION) -> ModelVersion | None:
    """The version holding `alias`, or None when nothing holds it."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code in NO_ALIAS_CODES:
            return None
        raise


def load_version(name: str, version: str) -> Any:
    """By explicit version, never by alias: an alias can move between lookup and load."""
    return mlflow.sklearn.load_model(f"models:/{name}/{version}")


def promote(client: MlflowClient, name: str, version: str) -> None:
    """Point `champion` at `version`, keeping the current champion as `previous_champion`
    so a rollback can restore it."""
    version = str(version)
    current = get_alias(client, name, CHAMPION)
    if current is not None and str(current.version) != version:
        client.set_registered_model_alias(name, PREVIOUS_CHAMPION, str(current.version))
    client.set_registered_model_alias(name, CHAMPION, version)


def rollback(client: MlflowClient, name: str) -> str:
    """Point `champion` back at `previous_champion` and return that version. The threshold
    is a tag on the version, so it comes back with it.

    `previous_champion` is then removed, so a second rollback fails instead of flipping
    back to the version just rolled back from.
    """
    previous = get_alias(client, name, PREVIOUS_CHAMPION)
    if previous is None:
        raise LookupError(f"{name} has no {PREVIOUS_CHAMPION} to roll back to")
    current = get_alias(client, name, CHAMPION)
    if current is not None:
        client.set_model_version_tag(name, str(current.version), "rolled_back_at", _now())
    client.set_registered_model_alias(name, CHAMPION, str(previous.version))
    client.delete_registered_model_alias(name, PREVIOUS_CHAMPION)
    log.info("%s: champion rolled back to version %s", name, previous.version)
    return str(previous.version)


def run_gate(
    client: MlflowClient,
    name: str,
    challenger: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_p_better: float = GATE_MIN_P_BETTER,
    n_bootstraps: int = GATE_BOOTSTRAPS,
) -> dict:
    """Decide whether version `challenger` replaces the champion, record why on the version,
    and promote it if it does. Returns the decision as a small dict (it travels via XCom).

    - No champion yet: promote.
    - The champion was judged on another test set, or on none (it predates the gate):
      the two can't be compared on equal terms, so promote, and say so.
    - Otherwise: score both on the test set, each at its own threshold, and promote only if
      the challenger's F2 is higher in at least `min_p_better` of the paired resamples.
    """
    challenger = str(challenger)
    mv = client.get_model_version(name, challenger)
    missing = [key for key in ("threshold", "test_split") if key not in mv.tags]
    if missing:
        raise ValueError(f"{name} v{challenger} lacks the tags {missing}; register() sets them")

    champion = get_alias(client, name, CHAMPION)
    result: dict = {"model": name, "version": challenger, "min_p_better": min_p_better}
    if champion is None:
        decision, reason = "promoted", "no champion yet"
    elif str(champion.version) == challenger:
        raise ValueError(f"{name} v{challenger} is already the champion")
    elif champion.tags.get("test_split") != mv.tags["test_split"]:
        decision = "promoted"
        reason = (
            f"champion v{champion.version} was not judged on test split "
            f"{mv.tags['test_split']} (its own: {champion.tags.get('test_split', 'none')}), "
            "so there is nothing to compare against"
        )
    else:
        stats = paired_bootstrap(
            np.asarray(y_test),
            _flags(name, str(champion.version), champion.tags, X_test),
            _flags(name, challenger, mv.tags, X_test),
            n=n_bootstraps,
        )
        accepted = stats["p_better"] >= min_p_better
        decision = "promoted" if accepted else "rejected"
        reason = (
            f"P(F2 better than v{champion.version}) = {stats['p_better']:.3f} "
            f"{'>=' if accepted else '<'} {min_p_better:.2f}"
        )
        result.update(
            p_better=stats["p_better"],
            mean_diff=stats["mean_diff"],
            ci90=list(stats["ci90"]),
            champion_f2=stats["champion_score"],
            challenger_f2=stats["challenger_score"],
        )
    if champion is not None:
        result["champion_version"] = str(champion.version)
    result.update(decision=decision, reason=reason, at=_now())

    for key, value in _gate_tags(result).items():
        client.set_model_version_tag(name, challenger, key, value)
    if decision == "promoted":
        promote(client, name, challenger)
    log.info("%s v%s %s: %s", name, challenger, decision, reason)
    return result


def _flags(name: str, version: str, tags: dict, X: pd.DataFrame) -> np.ndarray:
    """The version's high-risk flags on X, at the version's own threshold."""
    proba = load_version(name, version).predict_proba(X)[:, 1]
    return proba >= float(tags["threshold"])


def _gate_tags(result: dict) -> dict[str, str]:
    tags = {
        "gate_decision": result["decision"],
        "gate_reason": result["reason"],
        "gate_at": result["at"],
        "gate_min_p_better": str(result["min_p_better"]),
    }
    if "champion_version" in result:
        tags["gate_champion"] = result["champion_version"]
    if "p_better" in result:
        tags.update(
            gate_p_better=f"{result['p_better']:.4f}",
            gate_mean_diff=f"{result['mean_diff']:.4f}",
            gate_ci90=f"{result['ci90'][0]:.4f},{result['ci90'][1]:.4f}",
            gate_champion_f2=f"{result['champion_f2']:.4f}",
            gate_challenger_f2=f"{result['challenger_f2']:.4f}",
        )
    return tags


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
