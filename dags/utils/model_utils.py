import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

BETA = 2  # para podrr ajustar el f-score. Hay que reentrenar cada vez que se cambia
METRIC_NAME = f"F{BETA}"  # se usa como clave de métricas en dicts/tablas
f2_scorer = make_scorer(fbeta_score, beta=BETA)


def cv_f2(model, X, y, n_splits=5, shuffle=True):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring=f2_scorer, n_jobs=-1).mean()


class AgeBaselineClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(int)

    def predict_proba(self, X):
        # Score continuo = edad estandarizada mapeada a [0, 1] con sigmoide.
        # Preserva el ranking por edad para que AUC-ROC y PR-AUC midan
        # de verdad qué tan bien la edad sola separa las clases.
        p1 = expit(X[:, 0])
        return np.column_stack([1 - p1, p1])
