"""Shared constants for the lifecycle modules (stroke/ and model/train.py). Env-driven where a
deployment may want to change them, so the same code runs unchanged on the laptop and inside
containers. Only the standard library here: the Airflow DAG files import this module while
the scheduler parses them, so it must stay fast to import.
"""

import os

# The frozen split: one stratified 80/20 split, fixed by the seed.
RANDOM_SEED = 42
TEST_SIZE = 0.20

CV_FOLDS = 5
# F2: recall counts four times as much as precision. Missing a patient at risk costs more
# than a false alarm in a screening tool.
BETA = 2.0

# The model families trained on every run. age_only is the floor: a logistic regression on
# age alone, which any real model has to beat.
CANDIDATES = ("age_only", "logreg", "random_forest", "xgboost")

MODEL_NAME = os.getenv("MODEL_NAME", "stroke-model")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "stroke-risk")
CHAMPION, PREVIOUS_CHAMPION = "champion", "previous_champion"

# The promotion gate: a challenger replaces the champion only if its F2 beats the
# champion's in at least this share of paired bootstrap resamples of the test set.
GATE_MIN_P_BETTER = float(os.getenv("GATE_MIN_P_BETTER", "0.90"))
GATE_BOOTSTRAPS = 2000
