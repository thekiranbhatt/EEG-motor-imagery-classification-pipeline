"""Train and evaluate the left- vs. right-hand model."""
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import CLF_TMAX, CLF_TMIN, CV_FOLDS, N_CSP_COMPONENTS, RANDOM_STATE


def build_pipeline() -> Pipeline:
    """Build the CSP and LDA model."""
    return Pipeline([
        ("csp", CSP(n_components=N_CSP_COMPONENTS, log=True)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
def get_data(epochs):
    """Return trial data and labels for the imagery window."""
    cropped = epochs.copy().crop(tmin=CLF_TMIN, tmax=CLF_TMAX)
    X = cropped.get_data(copy=True)
    y = cropped.events[:, -1] - 2  # left -> 0, right -> 1
    return X, y


def evaluate(X, y):
    """Evaluate the model with cross-validation."""
    pipe = build_pipeline()
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_pred = np.empty_like(y)
    y_prob = np.empty(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        fold_model = build_pipeline()
        fold_model.fit(X_train, y_train)

        y_pred[test_idx] = fold_model.predict(X_test)
        y_prob[test_idx] = fold_model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "auc": roc_auc_score(y, y_prob),
        "chance_level": max(np.mean(y), 1 - np.mean(y)),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "report": classification_report(
            y,
            y_pred,
            target_names=["left", "right"],
            zero_division=0,
        ),
    }


def fit_final(X, y) -> Pipeline:
    """Fit the final model for the sensor-pattern plot."""
    return build_pipeline().fit(X, y)
