import numpy as np
import pandas as pd
import pytest
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


def test_temporal_split_assigns_every_row_at_most_once_and_is_ordered():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    result = temporal_train_val_test_split(X, y, feature_df["obs_end"])
    train_idx, val_idx, test_idx = result[-3:]

    assert len(train_idx) + len(val_idx) + len(test_idx) <= len(X)
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


def test_embargo_gap_enforced_between_train_and_validation():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    embargo_days = 90
    result = temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=embargo_days)
    train_idx, val_idx, test_idx = result[-3:]

    dates = feature_df["obs_end"]
    gap = (dates.iloc[val_idx].min() - dates.iloc[train_idx].max()).days
    assert gap > embargo_days


def test_embargo_gap_enforced_between_validation_and_test():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    embargo_days = 90
    result = temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=embargo_days)
    train_idx, val_idx, test_idx = result[-3:]

    dates = feature_df["obs_end"]
    gap = (dates.iloc[test_idx].min() - dates.iloc[val_idx].max()).days
    assert gap > embargo_days


def test_rows_within_embargo_window_are_purged_from_all_splits():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)
    embargo_days = 90

    no_embargo_result = temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=0)
    embargo_result = temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=embargo_days)

    n_no_embargo = sum(len(idx) for idx in no_embargo_result[-3:])
    n_embargo = sum(len(idx) for idx in embargo_result[-3:])

    assert n_embargo < n_no_embargo


def test_embargo_days_is_configurable():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    small_embargo = temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=30)
    large_embargo = temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=150)

    n_small = sum(len(idx) for idx in small_embargo[-3:])
    n_large = sum(len(idx) for idx in large_embargo[-3:])

    assert n_large < n_small


def test_negative_embargo_days_raises():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    with pytest.raises(ValueError):
        temporal_train_val_test_split(X, y, feature_df["obs_end"], embargo_days=-1)
