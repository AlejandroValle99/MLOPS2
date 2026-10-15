"""Population drift: how far a window of incoming patients moved from a baseline.

A baseline ("reference") summarizes one population, here one hospital's training rows:
decile bins and their proportions for numeric columns, category frequencies for
categorical ones. A window of events is binned the same way and compared with the
Population Stability Index (PSI):

    PSI = sum over bins of (actual - expected) * ln(actual / expected)

Rules of thumb: below 0.1 stable, 0.1 to 0.25 a moderate shift, above 0.25 a major one.

Only numpy and pandas here: the stream worker imports this module, and its image has no
scikit-learn.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["work_type", "smoking_status"]
MONITORED = NUMERIC + CATEGORICAL

N_BINS = 10  # deciles
EPS = 1e-4  # stands in for an empty bin's proportion, which would otherwise give ln(0)


def _bin_proportions(values: np.ndarray, inner_edges: list[float]) -> np.ndarray:
    """Proportion of `values` in each bin. The outer bins are open-ended, so values outside
    the baseline's range still land somewhere instead of being dropped."""
    bins = np.searchsorted(inner_edges, values, side="right")
    counts = np.bincount(bins, minlength=len(inner_edges) + 1)
    return counts / counts.sum()


def build_reference(X: pd.DataFrame) -> dict:
    """The baseline for one population. Plain lists, floats and dicts, so it is JSON."""
    numeric = {}
    for col in NUMERIC:
        values = X[col].dropna().to_numpy(dtype=float)  # a missing bmi is not a value
        cuts = np.quantile(values, np.linspace(0, 1, N_BINS + 1))[1:-1]
        inner = np.unique(cuts).tolist()  # ties collapse duplicate cut points
        numeric[col] = {
            "inner_edges": inner,
            "proportions": _bin_proportions(values, inner).tolist(),
        }
    categorical = {
        col: {str(k): float(v) for k, v in X[col].value_counts(normalize=True).items()}
        for col in CATEGORICAL
    }
    return {"rows": len(X), "numeric": numeric, "categorical": categorical}


def psi(expected: ArrayLike, actual: ArrayLike, eps: float = EPS) -> float:
    """Population Stability Index between two matching arrays of bin proportions."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: dict, window: pd.DataFrame, min_count: int) -> dict[str, float]:
    """PSI per monitored column for one window of rows. {} when the window holds fewer than
    `min_count` rows: PSI on a handful of events is sampling noise, not drift."""
    if len(window) < min_count:
        return {}
    report = {}
    for col, ref in reference["numeric"].items():
        values = window[col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue  # nothing to compare (every bmi in the window missing)
        report[col] = psi(ref["proportions"], _bin_proportions(values, ref["inner_edges"]))
    for col, ref in reference["categorical"].items():
        actual = window[col].value_counts(normalize=True)
        # The union, so a category the baseline never saw counts as drift too.
        categories = list(ref) + [c for c in actual.index if c not in ref]
        report[col] = psi(
            [ref.get(c, 0.0) for c in categories], [actual.get(c, 0.0) for c in categories]
        )
    return report


def build_hospital_references(X: pd.DataFrame, hospitals: pd.Series) -> dict[str, dict]:
    """One baseline per hospital, from that hospital's own rows. The hospitals differ by
    design (age is part of the rule), so one baseline for everyone would flag age "drift"
    at every hospital forever."""
    return {str(h): build_reference(X[hospitals == h]) for h in sorted(hospitals.unique())}
