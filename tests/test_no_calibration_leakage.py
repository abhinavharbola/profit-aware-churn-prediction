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


def test_final_model_is_fit_only_on_the_train_split():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    original_fit = XGBClassifier.fit
    fit_calls = []

    def spy_fit(self, X_arg, y_arg, *args, **kwargs):
        fit_calls.append((len(X_arg), "eval_set" in kwargs and kwargs["eval_set"] is not None))
        return original_fit(self, X_arg, y_arg, *args, **kwargs)

    with patch.object(XGBClassifier, "fit", spy_fit):
        result = train_model(X, y, feature_cols, feature_df["obs_end"], n_trials=3)

    X_train, X_val = result[3], result[5]

    final_fit_row_count, final_fit_had_eval_set = fit_calls[-1]

    assert final_fit_row_count == len(X_train)
    assert final_fit_row_count != len(X_train) + len(X_val)
    assert final_fit_had_eval_set is False


def test_train_and_validation_indices_returned_by_train_model_are_disjoint():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    result = train_model(X, y, feature_cols, feature_df["obs_end"], n_trials=3)
    train_idx, val_idx = result[-3], result[-2]

    assert len(set(train_idx) & set(val_idx)) == 0
