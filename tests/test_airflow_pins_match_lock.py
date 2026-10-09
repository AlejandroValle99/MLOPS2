"""D3 (docs/PLAN.md): docker/airflow/requirements.txt pins the libraries that matter for
pickle compatibility (the model is pickled in the Airflow image, unpickled in the serving
image) to the exact versions uv.lock resolved. This test fails the moment those drift —
cheaper than finding out via a broken mlflow.sklearn.load_model in a container.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
REQUIREMENTS = ROOT / "docker/airflow/requirements.txt"
LOCK = ROOT / "uv.lock"

# Only the libraries requirements.txt pins with "==" (psycopg2-binary uses ">=" on
# purpose — it isn't involved in pickling a model, so it doesn't need to match exactly).
PINNED_LIBS = [
    "scikit-learn",
    "xgboost",
    "mlflow",
    "numpy",
    "pandas",
    "boto3",
    "optuna",
    "matplotlib",
]


def _pins_from_requirements() -> dict[str, str]:
    text = REQUIREMENTS.read_text()
    pins = {}
    for lib in PINNED_LIBS:
        match = re.search(rf"^{re.escape(lib)}==([^\s#]+)", text, re.MULTILINE)
        assert match, f"{lib} is not pinned with == in {REQUIREMENTS}"
        pins[lib] = match.group(1)
    return pins


def _versions_from_lock() -> dict[str, str]:
    text = LOCK.read_text()
    versions = {}
    for lib in PINNED_LIBS:
        match = re.search(rf'^name = "{re.escape(lib)}"\nversion = "([^"]+)"', text, re.MULTILINE)
        assert match, f"{lib} not found in {LOCK}"
        versions[lib] = match.group(1)
    return versions


def test_airflow_requirements_match_uv_lock_exactly():
    pinned = _pins_from_requirements()
    locked = _versions_from_lock()
    mismatches = {
        lib: (pinned[lib], locked[lib]) for lib in PINNED_LIBS if pinned[lib] != locked[lib]
    }
    assert not mismatches, (
        "docker/airflow/requirements.txt has drifted from uv.lock "
        f"(library: (requirements.txt, uv.lock)): {mismatches}. "
        "Update docker/airflow/requirements.txt's == pins to match uv.lock."
    )
