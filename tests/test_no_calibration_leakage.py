import numpy as np
import pandas as pd
from unittest.mock import patch
from xgboost import XGBClassifier
from src.modeling.trainer import prepare_data, train_model


def _synthetic_feature_df():
    rows = []
    start = pd.Timestamp("2010-01-01")
    rng = np.random.RandomState(0)
    for window_id in range(30):
        obs_end = start + pd.Timedelta(days=30 * window_id)
        for customer_id in range(10):
            rows.append({
                "customer_id": customer_id,
                "window_id": window_id,
                "obs_end": obs_end,
                "recency": rng.randint(0, 100),
                "frequency": rng.randint(1, 10),
                "monetary_total": rng.uniform(10, 500),
                "churn": rng.randint(0, 2),
            })
    return pd.DataFrame(rows)


def _train(n_trials=3):
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    return train_model(X, y, feature_cols, feature_df["obs_end"], groups, n_trials=n_trials)


def test_every_fit_uses_only_train_rows_and_final_fit_has_no_eval_set():
    original_fit = XGBClassifier.fit
    fit_calls = []

    def spy_fit(self, X_arg, y_arg, *args, **kwargs):
        eval_set = kwargs.get("eval_set")
        eval_rows = len(eval_set[0][0]) if eval_set else None
        fit_calls.append((len(X_arg), eval_rows))
        return original_fit(self, X_arg, y_arg, *args, **kwargs)

    with patch.object(XGBClassifier, "fit", spy_fit):
        result = _train()

    assert len(fit_calls) >= 2
    for fit_rows, _ in fit_calls:
        assert fit_rows == len(result.X_train)
    for _, eval_rows in fit_calls[:-1]:
        assert eval_rows == len(result.X_tune)
    assert fit_calls[-1][1] is None
    assert sum(1 for _, eval_rows in fit_calls if eval_rows is None) == 1
    assert result.final_fit_rows == len(result.X_train)


def test_train_tune_calibration_and_test_indices_are_pairwise_disjoint():
    result = _train()
    sets = [set(result.train_idx), set(result.tune_idx), set(result.cal_idx), set(result.test_idx)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])


def test_tuning_and_calibration_parts_share_no_customers():
    feature_df = _synthetic_feature_df()
    result = _train()
    tune_customers = set(feature_df.iloc[result.tune_idx]["customer_id"])
    cal_customers = set(feature_df.iloc[result.cal_idx]["customer_id"])
    assert tune_customers.isdisjoint(cal_customers)
    assert len(tune_customers) > 0 and len(cal_customers) > 0


def test_tuning_and_calibration_parts_are_both_inside_the_validation_period():
    feature_df = _synthetic_feature_df()
    result = _train()
    dates = feature_df["obs_end"]
    train_max = dates.iloc[result.train_idx].max()
    test_min = dates.iloc[result.test_idx].min()
    for idx in (result.tune_idx, result.cal_idx):
        assert dates.iloc[idx].min() > train_max
        assert dates.iloc[idx].max() < test_min
