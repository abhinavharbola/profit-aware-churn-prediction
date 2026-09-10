import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from config import CALIBRATION_METHOD


def calibrate_probabilities(model, X_calibration, y_calibration, method=CALIBRATION_METHOD):
    raw_scores = model.predict_proba(X_calibration)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_scores, y_calibration)

        def calibration_func(scores):
            return calibrator.transform(scores)

    elif method == "platt":
        calibrator = LogisticRegression()
        calibrator.fit(raw_scores.reshape(-1, 1), y_calibration)

        def calibration_func(scores):
            return calibrator.predict_proba(np.array(scores).reshape(-1, 1))[:, 1]

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return calibration_func, calibrator



