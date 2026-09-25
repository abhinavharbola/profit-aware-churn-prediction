import numpy as np
import pandas as pd
from src.modeling.trainer import prepare_data, temporal_train_val_test_split


def _synthetic_feature_df():
    rows = []
    start = pd.Timestamp("2010-01-01")
    for window_id in range(30):
        obs_end = start + pd.Timedelta(days=30 * window_id)
        for customer_id in range(10):
            rows.append({
                "customer_id": customer_id,
                "window_id": window_id,
                "obs_end": obs_end,
                "recency": np.random.randint(0, 100),
                "frequency": np.random.randint(1, 10),
                "monetary_total": np.random.uniform(10, 500),
                "churn": np.random.randint(0, 2),
            })
    return pd.DataFrame(rows)


def test_temporal_split_assigns_every_row_once_and_is_ordered():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    result = temporal_train_val_test_split(X, y, feature_df["obs_end"])
    train_idx, val_idx, test_idx = result[-3:]

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(X)
    assert len(set(np.concatenate([train_idx, val_idx, test_idx]))) == len(train_idx) + len(val_idx) + len(test_idx)

    dates = feature_df["obs_end"]
    assert dates.iloc[train_idx].max() < dates.iloc[val_idx].min()
    assert dates.iloc[val_idx].max() < dates.iloc[test_idx].min()


def test_temporal_split_has_no_row_overlap():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    result = temporal_train_val_test_split(X, y, feature_df["obs_end"])
    train_idx, val_idx, test_idx = result[-3:]

    memberships = np.concatenate([train_idx, val_idx, test_idx])
    assert len(set(memberships)) == len(memberships)


def test_feature_columns_exclude_identifiers():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    assert "customer_id" not in feature_cols
    assert "churn" not in feature_cols
    assert "window_id" not in feature_cols
    assert "obs_end" not in feature_cols
