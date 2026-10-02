import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from config import CALIBRATION_METHOD, CALIBRATION_FOLDS, RANDOM_SEED


def fit_calibrator(raw_scores, y, method=CALIBRATION_METHOD):
    raw_scores = np.asarray(raw_scores)
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_scores, y)
    elif method == "platt":
        calibrator = LogisticRegression()
        calibrator.fit(raw_scores.reshape(-1, 1), y)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return calibrator


def make_calibration_func(calibrator, method=CALIBRATION_METHOD):
    if method == "isotonic":
        def calibration_func(scores):
            return calibrator.transform(np.asarray(scores))
    elif method == "platt":
        def calibration_func(scores):
            return calibrator.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return calibration_func


def calibrate_probabilities(model, X_calibration, y_calibration, method=CALIBRATION_METHOD):
    raw_scores = model.predict_proba(X_calibration)[:, 1]
    calibrator = fit_calibrator(raw_scores, y_calibration, method)
    return make_calibration_func(calibrator, method), calibrator


def cross_fitted_probabilities(raw_scores, y, method=CALIBRATION_METHOD,
                               n_splits=CALIBRATION_FOLDS, seed=RANDOM_SEED):
    raw_scores = np.asarray(raw_scores)
    labels = np.asarray(y)
    calibrated = np.zeros(len(raw_scores), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in splitter.split(raw_scores, labels):
        calibrator = fit_calibrator(raw_scores[fit_idx], labels[fit_idx], method)
        calibrated[hold_idx] = make_calibration_func(calibrator, method)(raw_scores[hold_idx])
    return calibrated
