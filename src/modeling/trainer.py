import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import average_precision_score
from config import (
    RANDOM_SEED, OPTUNA_TRIALS, VALIDATION_SIZE, TEST_SIZE,
    EARLY_STOPPING_ROUNDS, EMBARGO_DAYS
)


def prepare_data(feature_df):
    feature_cols = [c for c in feature_df.columns if c not in
                    ("customer_id", "churn", "window_id", "obs_end")]
    X = feature_df[feature_cols].copy()
    y = feature_df["churn"].copy()
    groups = feature_df["customer_id"].copy()
    return X, y, groups, feature_cols


def temporal_train_val_test_split(
    X, y, obs_end, validation_size=VALIDATION_SIZE,
    test_size=TEST_SIZE, embargo_days=EMBARGO_DAYS
):
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("Validation and test sizes must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("Validation and test sizes must sum to less than 1.")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative.")

    dates = obs_end.reset_index(drop=True)
    unique_dates = sorted(dates.dropna().unique())
    n = len(unique_dates)
    if n < 6:
        raise ValueError("At least 6 distinct observation dates are required for a temporal split.")

    span_days = (pd.Timestamp(unique_dates[-1]) - pd.Timestamp(unique_dates[0])).days
    typical_gap_days = span_days / (n - 1) if span_days > 0 else 0

    required_step = int(np.ceil(embargo_days / typical_gap_days)) if typical_gap_days > 0 and embargo_days > 0 else 0
    required_purge = max(0, required_step - 1)
    purge = min(required_purge, max(0, (n - 3) // 2))
    embargo_reduced = purge < required_purge
    if embargo_reduced:
        warnings.warn(
            f"A {embargo_days}-day embargo needs {required_purge} purged dates per "
            f"boundary but only {n} distinct observation dates exist; reduced to "
            f"{purge}. Label periods across split boundaries may still overlap."
        )
    step = purge + 1

    n_effective = n - 2 * purge

    train_fraction = 1 - validation_size - test_size
    train_count = min(n_effective - 2, max(1, int(round(n_effective * train_fraction))))
    val_count = min(n_effective - train_count - 1, max(1, int(round(n_effective * validation_size))))

    train_boundary = train_count - 1
    val_start_boundary = train_boundary + step
    val_boundary = val_start_boundary + val_count - 1
    test_start_boundary = val_boundary + step

    train_end = unique_dates[train_boundary]
    val_start_date = unique_dates[val_start_boundary]
    val_end = unique_dates[val_boundary]
    test_start_date = unique_dates[test_start_boundary]

    train_mask = dates <= train_end
    val_mask = (dates >= val_start_date) & (dates <= val_end)
    test_mask = dates >= test_start_date
    masks = [train_mask, val_mask, test_mask]

    if any(mask.sum() == 0 for mask in masks):
        counts = [int(mask.sum()) for mask in masks]
        raise ValueError(
            "Temporal split produced an empty partition after reserving the "
            f"{embargo_days}-day embargo. Increase the available history, "
            f"reduce validation_size/test_size, or reduce embargo_days. "
            f"Counts: {counts}"
        )

    train_idx = dates.index[train_mask].to_numpy()
    val_idx = dates.index[val_mask].to_numpy()
    test_idx = dates.index[test_mask].to_numpy()

    train_dates = dates.iloc[train_idx]
    val_dates = dates.iloc[val_idx]
    test_dates = dates.iloc[test_idx]
    gap_train_val = (val_dates.min() - train_dates.max()).days
    gap_val_test = (test_dates.min() - val_dates.max()).days
    if not embargo_reduced and (gap_train_val < embargo_days or gap_val_test < embargo_days):
        raise AssertionError(
            f"Temporal split gaps ({gap_train_val}, {gap_val_test} days) are below the "
            f"required {embargo_days}-day embargo."
        )

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        train_idx, val_idx, test_idx
    )


def split_validation_by_customer(val_idx, groups):
    customer_ids = groups.iloc[val_idx].to_numpy()
    tune_positions = np.where(customer_ids % 2 == 0)[0]
    calibration_positions = np.where(customer_ids % 2 == 1)[0]
    if len(tune_positions) == 0 or len(calibration_positions) == 0:
        raise ValueError("Validation period has too few distinct customers to split into tuning and calibration parts.")
    return tune_positions, calibration_positions


def objective(trial, X_train, y_train, X_tune, y_tune):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "eval_metric": "aucpr",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_tune, y_tune)], verbose=False)
    y_pred = model.predict_proba(X_tune)[:, 1]
    trial.set_user_attr("best_iteration", model.best_iteration)
    return average_precision_score(y_tune, y_pred)


@dataclass
class TrainingResult:
    model: object
    study: object
    feature_cols: list
    X_train: pd.DataFrame
    y_train: pd.Series
    X_tune: pd.DataFrame
    y_tune: pd.Series
    X_cal: pd.DataFrame
    y_cal: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_idx: np.ndarray
    tune_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray
    final_fit_rows: int


def train_model(X, y, feature_cols, obs_end, groups, n_trials=OPTUNA_TRIALS):
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
        temporal_train_val_test_split(X, y, obs_end)

    n_purged = len(X) - (len(train_idx) + len(val_idx) + len(test_idx))
    if n_purged > 0:
        print(f"Embargo purge: {n_purged} rows near split boundaries excluded from all splits.")

    tune_pos, cal_pos = split_validation_by_customer(val_idx, groups)
    X_tune, y_tune = X_val.iloc[tune_pos].reset_index(drop=True), y_val.iloc[tune_pos].reset_index(drop=True)
    X_cal, y_cal = X_val.iloc[cal_pos].reset_index(drop=True), y_val.iloc[cal_pos].reset_index(drop=True)
    tune_idx, cal_idx = val_idx[tune_pos], val_idx[cal_pos]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_tune, y_tune),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = dict(study.best_params)
    best_iteration = study.best_trial.user_attrs.get("best_iteration")
    if best_iteration is not None:
        best_params["n_estimators"] = best_iteration + 1
    best_params.update({
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    return TrainingResult(
        model=model, study=study, feature_cols=feature_cols,
        X_train=X_train, y_train=y_train, X_tune=X_tune, y_tune=y_tune,
        X_cal=X_cal, y_cal=y_cal, X_test=X_test, y_test=y_test,
        train_idx=train_idx, tune_idx=tune_idx, cal_idx=cal_idx, test_idx=test_idx,
        final_fit_rows=len(X_train)
    )
