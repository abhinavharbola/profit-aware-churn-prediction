import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from src.modeling.calibrator import (
    fit_calibrator, make_calibration_func, cross_fitted_probabilities
)
from src.evaluation.metrics import compute_metrics


def _scores_and_labels():
    rng = np.random.RandomState(1)
    raw = rng.uniform(0, 1, 400)
    labels = (rng.uniform(0, 1, 400) < raw ** 2).astype(int)
    return raw, labels


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_func_returns_probabilities(method):
    raw, labels = _scores_and_labels()
    func = make_calibration_func(fit_calibrator(raw, labels, method), method)
    out = func(raw)
    assert out.shape == raw.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_unknown_method_raises():
    raw, labels = _scores_and_labels()
    with pytest.raises(ValueError):
        fit_calibrator(raw, labels, "bogus")


def test_cross_fitted_probabilities_are_out_of_fold_not_in_sample():
    raw, labels = _scores_and_labels()
    in_sample = make_calibration_func(fit_calibrator(raw, labels, "isotonic"), "isotonic")(raw)
    oof = cross_fitted_probabilities(raw, labels, method="isotonic", n_splits=5)
    assert oof.shape == raw.shape
    assert not np.allclose(oof, in_sample)
    in_sample_brier = np.mean((in_sample - labels) ** 2)
    oof_brier = np.mean((oof - labels) ** 2)
    assert oof_brier >= in_sample_brier


def test_cross_fitted_probabilities_are_deterministic():
    raw, labels = _scores_and_labels()
    a = cross_fitted_probabilities(raw, labels, method="isotonic")
    b = cross_fitted_probabilities(raw, labels, method="isotonic")
    assert np.array_equal(a, b)


def test_pr_auc_is_computed_on_raw_scores_and_brier_on_calibrated():
    raw, labels = _scores_and_labels()
    calibrated = make_calibration_func(fit_calibrator(raw, labels, "isotonic"), "isotonic")(raw)
    metrics = compute_metrics(labels, raw, calibrated)
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(labels, raw))
    assert metrics["brier_score"] == pytest.approx(np.mean((calibrated - labels) ** 2))
