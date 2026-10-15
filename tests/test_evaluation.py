"""stroke.evaluation: metrics from counts, the threshold search, calibration error and the
paired bootstrap behind the promotion gate."""

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from stroke.evaluation import (
    best_threshold,
    confusion_counts,
    expected_calibration_error,
    fbeta_from_counts,
    metrics_at,
    paired_bootstrap,
)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_fbeta_from_counts_matches_sklearn(beta):
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    tp, fp, fn = confusion_counts(y_true, y_pred)
    expected = fbeta_score(y_true, y_pred, beta=beta)
    assert fbeta_from_counts(tp, fp, fn, beta=beta) == pytest.approx(expected)


def test_fbeta_is_zero_when_there_is_nothing_to_score():
    assert fbeta_from_counts(0, 0, 0) == 0.0
    assert fbeta_from_counts(0, 5, 3) == 0.0


def test_confusion_counts():
    assert confusion_counts([1, 1, 0, 0, 1], [1, 0, 1, 0, 1]) == (2, 1, 1)


def test_the_threshold_finder_picks_the_separating_cut():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    assert best_threshold(y, proba) == (0.7, 1.0)


def test_the_threshold_finder_matches_a_brute_force_search():
    rng = np.random.default_rng(1)
    y = rng.random(400) < 0.1
    proba = np.round(np.clip(rng.normal(0.2 + 0.3 * y, 0.15), 0, 1), 2)  # rounding: many ties
    brute = {t: fbeta_score(y, proba >= t, beta=2, zero_division=0) for t in np.unique(proba)}
    best = max(brute.values())
    threshold, score = best_threshold(y, proba)
    assert score == pytest.approx(best)
    assert threshold == max(t for t, f in brute.items() if f == pytest.approx(best))


def test_tied_probabilities_are_never_split_by_the_cut():
    # The two 0.5s are one negative and one positive: a cut "between" them doesn't exist
    # at serving time, where both rows are flagged or neither is.
    y = np.array([0, 0, 1, 0, 1])
    proba = np.array([0.1, 0.5, 0.5, 0.2, 0.9])
    threshold, score = best_threshold(y, proba)
    tp, fp, fn = confusion_counts(y, proba >= threshold)
    assert score == pytest.approx(fbeta_from_counts(tp, fp, fn))


def test_metrics_at_agrees_with_sklearn():
    rng = np.random.default_rng(2)
    y = (rng.random(1000) < 0.1).astype(int)
    proba = np.clip(rng.normal(0.1 + 0.3 * y, 0.1), 0, 1)
    m = metrics_at(y, proba, threshold=0.2)
    flagged = proba >= 0.2
    assert m["f2"] == pytest.approx(fbeta_score(y, flagged, beta=2))
    assert m["recall"] == pytest.approx(recall_score(y, flagged))
    assert m["precision"] == pytest.approx(precision_score(y, flagged))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, proba))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, proba))
    assert m["brier"] == pytest.approx(brier_score_loss(y, proba))
    assert m["alert_rate"] == pytest.approx(flagged.mean())
    assert m["threshold"] == 0.2


def test_calibration_error_is_small_for_scores_drawn_from_their_own_probabilities():
    rng = np.random.default_rng(3)
    proba = rng.random(20_000)
    y = rng.random(20_000) < proba  # each outcome happens with exactly its predicted chance
    assert expected_calibration_error(y, proba) < 0.02


def test_calibration_error_is_large_for_inflated_probabilities():
    rng = np.random.default_rng(4)
    proba = rng.random(20_000) * 0.2
    y = rng.random(20_000) < proba
    inflated = np.clip(proba * 3, 0, 1)  # what class weights do to a model's scores
    assert expected_calibration_error(y, inflated) > 0.1


def test_the_bootstrap_gives_zero_for_identical_models():
    rng = np.random.default_rng(5)
    y = rng.random(1000) < 0.05
    pred = rng.random(1000) < 0.2
    result = paired_bootstrap(y, pred, pred, n=500)
    assert result["p_better"] == 0  # a tie is not a win
    assert result["mean_diff"] == 0
    assert result["ci90"] == (0, 0)


def test_the_bootstrap_is_confident_about_a_clearly_better_model():
    rng = np.random.default_rng(6)
    y = rng.random(1000) < 0.05
    weak = rng.random(1000) < 0.2  # flags at random
    strong = y | (rng.random(1000) < 0.1)  # catches every positive, few false alarms
    result = paired_bootstrap(y, weak, strong, n=500)
    assert result["p_better"] > 0.95
    assert result["ci90"][0] > 0
    assert result["challenger_score"] > result["champion_score"]
    assert paired_bootstrap(y, strong, weak, n=500)["p_better"] < 0.05


def test_every_bootstrap_resample_keeps_the_test_sets_positives():
    """One positive in 20 rows: a model that flags exactly that patient beats one that flags
    nobody in every resample that contains the patient. Resampling positives separately
    keeps them in every resample; a plain resample would leave the patient out about a third
    of the time ((19/20)^20 = 0.36), and those resamples would be ties."""
    y = np.zeros(20, dtype=bool)
    y[0] = True
    result = paired_bootstrap(y, np.zeros(20, dtype=bool), y, n=500)
    assert result["p_better"] == 1.0


def test_the_bootstrap_is_reproducible_with_its_seed():
    rng = np.random.default_rng(7)
    y = rng.random(300) < 0.1
    a, b = rng.random(300) < 0.2, rng.random(300) < 0.2
    assert paired_bootstrap(y, a, b, n=200) == paired_bootstrap(y, a, b, n=200)


def test_the_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([0, 1, 0], [0, 1], [0, 1, 1])
