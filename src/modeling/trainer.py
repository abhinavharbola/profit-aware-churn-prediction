import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import average_precision_score
from config import (
    RANDOM_SEED, OPTUNA_TRIALS, VALIDATION_SIZE, TEST_SIZE
)


def prepare_data(feature_df):
    feature_cols = [c for c in feature_df.columns if c not in
                    ("customer_id", "churn", "window_id", "obs_end")]
    X = feature_df[feature_cols].copy()
    y = feature_df["churn"].copy()
    groups = feature_df["customer_id"].copy()
    return X, y, groups, feature_cols


def temporal_train_val_test_split(
    X, y, groups, obs_end, validation_size=VALIDATION_SIZE,
    test_size=TEST_SIZE
):
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("Validation and test sizes must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("Validation and test sizes must sum to less than 1.")

    dates = obs_end.reset_index(drop=True)
    unique_dates = sorted(dates.dropna().unique())
    if len(unique_dates) < 6:
        raise ValueError("At least 6 distinct observation dates are required for a temporal split.")

    n = len(unique_dates)
    train_fraction = 1 - validation_size - test_size
    train_boundary = min(n - 2, max(1, int(round(n * train_fraction)) - 1))
    val_boundary = min(n - 1, max(train_boundary + 1, int(round(n * (1 - test_size))) - 1))

    train_end = unique_dates[train_boundary]
    val_end = unique_dates[val_boundary]

    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    masks = [train_mask, val_mask, test_mask]

    if any(mask.sum() == 0 for mask in masks):
        counts = [int(mask.sum()) for mask in masks]
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"Increase the available history or reduce purge/split sizes. Counts: {counts}"
        )

    train_idx = dates.index[train_mask].to_numpy()
    val_idx = dates.index[val_mask].to_numpy()
    test_idx = dates.index[test_mask].to_numpy()

    train_dates = dates.iloc[train_idx]
    val_dates = dates.iloc[val_idx]
    test_dates = dates.iloc[test_idx]
    if not (train_dates.max() < val_dates.min() and val_dates.max() < test_dates.min()):
        raise AssertionError("Temporal split is not strictly chronological.")

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        train_idx, val_idx, test_idx
    )



def objective(trial, X_train, y_train, X_val, y_val):
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
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_pred)


def train_model(X, y, groups, feature_cols, obs_end):
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
        temporal_train_val_test_split(X, y, groups, obs_end)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_params.update({
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    })

    model = xgb.XGBClassifier(**best_params)
    X_final = pd.concat([X_train, X_val], ignore_index=True)
    y_final = pd.concat([y_train, y_val], ignore_index=True)
    model.fit(X_final, y_final, verbose=False)

    return model, study, feature_cols, X_val, y_val, X_test, y_test, val_idx, test_idx

