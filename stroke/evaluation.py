"""Model evaluation: metrics at an operating point, the threshold search, calibration error,
and the statistics behind the promotion gate. numpy and scikit-learn's metrics only, no
MLflow, so every function here is tested on plain arrays.

F-beta is computed from counts, (1 + b²)·TP / ((1 + b²)·TP + b²·FN + FP), instead of calling
scikit-learn per resample: the gate scores two models on thousands of resamples, and the
threshold search scores every distinct probability as a cut.
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from stroke.config import BETA, GATE_BOOTSTRAPS, RANDOM_SEED

# The metric that picks the threshold, the finalist and the gate's winner ("f2").
PRIMARY = f"f{BETA:g}"


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int]:
    """(tp, fp, fn): all that F-beta needs."""
    actual = np.asarray(y_true).astype(bool)
    predicted = np.asarray(y_pred).astype(bool)
    return (
        int(np.sum(actual & predicted)),
        int(np.sum(~actual & predicted)),
        int(np.sum(actual & ~predicted)),
    )


def _fbeta(tp: ArrayLike, fp: ArrayLike, fn: ArrayLike, beta: float) -> np.ndarray:
    """F-beta for arrays of counts at once; 0 where there is nothing to score."""
    b2 = beta**2
    numerator = (1 + b2) * np.asarray(tp, dtype=float)
    denominator = numerator + b2 * np.asarray(fn, dtype=float) + np.asarray(fp, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = BETA) -> float:
    """F-beta from the confusion counts. 0.0 when there are no positives and no alerts."""
    return float(_fbeta(tp, fp, fn, beta))


def best_threshold(y_true: ArrayLike, proba: ArrayLike, beta: float = BETA) -> tuple[float, float]:
    """(threshold, F-beta) maximizing F-beta, where a row is flagged when proba >= threshold.

    Every distinct probability is a candidate cut. Sorting once and counting cumulatively
    scores all of them in one pass. Among cuts with the same F-beta, the highest wins: the
    same recall and precision with fewer alerts.
    """
    actual = np.asarray(y_true).astype(bool)
    scores = np.asarray(proba, dtype=float)
    order = np.argsort(-scores, kind="stable")
    scores, actual = scores[order], actual[order]
    tp = np.cumsum(actual)  # positives flagged if the cut sits at this row
    fp = np.cumsum(~actual)
    # Cut only at the last row of each run of equal probabilities: tied rows are always
    # flagged together, as `proba >= threshold` will flag them at serving time.
    cuts = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    f = _fbeta(tp[cuts], fp[cuts], actual.sum() - tp[cuts], beta)
    best = int(np.argmax(f))  # the first maximum, so the highest threshold among ties
    return float(scores[cuts[best]]), float(f[best])


def expected_calibration_error(y_true: ArrayLike, proba: ArrayLike, n_bins: int = 10) -> float:
    """The average gap between predicted and observed frequency: predictions are grouped
    into equal-width probability bins, and each bin's gap counts by its share of rows."""
    actual = np.asarray(y_true, dtype=float)
    scores = np.asarray(proba, dtype=float)
    bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)  # 1.0 joins the last bin
    predicted_sum = np.bincount(bins, weights=scores, minlength=n_bins)
    observed_sum = np.bincount(bins, weights=actual, minlength=n_bins)
    # sum over bins of (rows_b / N) * |mean predicted_b - mean observed_b|
    return float(np.abs(predicted_sum - observed_sum).sum() / len(scores))


def metrics_at(
    y_true: ArrayLike, proba: ArrayLike, threshold: float, beta: float = BETA
) -> dict[str, float]:
    """Everything logged for one set of predictions at one operating point. Runs log these
    keys with an `oof_` or `test_` prefix."""
    actual = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    flagged = scores >= threshold
    tp, fp, fn = confusion_counts(actual, flagged)
    return {
        f"f{beta:g}": fbeta_from_counts(tp, fp, fn, beta),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "roc_auc": float(roc_auc_score(actual, scores)),
        "pr_auc": float(average_precision_score(actual, scores)),
        "brier": float(brier_score_loss(actual, scores)),
        "ece": expected_calibration_error(actual, scores),
        "threshold": float(threshold),
        "alert_rate": float(flagged.mean()),  # the share of patients flagged high risk
    }


def paired_bootstrap(
    y_true: ArrayLike,
    pred_champion: ArrayLike,
    pred_challenger: ArrayLike,
    n: int = GATE_BOOTSTRAPS,
    beta: float = BETA,
    seed: int = RANDOM_SEED,
) -> dict:
    """How often the challenger's F-beta beats the champion's over `n` resamples of the
    test set. pred_* are 0/1 predictions, each model at its OWN threshold.

    Paired: in every resample both models are scored on the same rows, so the luck of which
    patients were drawn affects both alike and only the difference between them is left.
    Stratified: positives and negatives are resampled separately, so every resample keeps
    the test set's number of positives (about 50 here; a plain resample could draw 35 or 65).
    """
    actual = np.asarray(y_true).astype(bool)
    champion = np.asarray(pred_champion).astype(bool)
    challenger = np.asarray(pred_challenger).astype(bool)
    if not len(actual) == len(champion) == len(challenger):
        raise ValueError("y_true and both predictions must have the same length")

    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(actual), np.flatnonzero(~actual)
    diffs = np.empty(n)
    for i in range(n):
        rows = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        f_champion = fbeta_from_counts(*confusion_counts(actual[rows], champion[rows]), beta)
        f_challenger = fbeta_from_counts(*confusion_counts(actual[rows], challenger[rows]), beta)
        diffs[i] = f_challenger - f_champion
    low, high = np.percentile(diffs, [5, 95])
    return {
        "p_better": float(np.mean(diffs > 0)),  # strictly better: a tie is not a win
        "mean_diff": float(diffs.mean()),
        "ci90": (float(low), float(high)),
        # On the whole test set, for the record
        "champion_score": fbeta_from_counts(*confusion_counts(actual, champion), beta),
        "challenger_score": fbeta_from_counts(*confusion_counts(actual, challenger), beta),
        "n": n,
    }
