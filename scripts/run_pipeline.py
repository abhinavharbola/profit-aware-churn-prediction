import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DIR, ARTIFACTS_DIR, DEFAULT_THRESHOLD,
    RANDOM_TARGET_FRACTION, CALIBRATION_METHOD, EMBARGO_DAYS
)
from src.data.cleaner import run_cleaning
from src.data.temporal import generate_windows
from src.features.rfm_engineer import build_feature_matrix, build_current_snapshot
from src.modeling.trainer import prepare_data, train_model
from src.modeling.calibrator import calibrate_probabilities, cross_fitted_probabilities
from src.evaluation.metrics import compute_metrics
from src.evaluation.profit_optimizer import (
    find_optimal_threshold,
    evaluate_random_baseline,
    evaluate_threshold_strategy,
    evaluate_expected_value_strategy,
    compute_avg_monthly_spend
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("=== 1. Cleaning raw data ===")
df_clean = run_cleaning()
print(f"Cleaned transactions: {len(df_clean)}")

print("=== 2. Generating temporal windows ===")
windows = generate_windows(df_clean)
print(f"Windows generated: {len(windows)}")

print("=== 3. Building feature matrix ===")
feature_df = build_feature_matrix(windows)
print(f"Feature matrix shape: {feature_df.shape}")
print(f"Churn rate: {feature_df['churn'].mean():.3f}")
feature_df.to_pickle(os.path.join(PROCESSED_DIR, "feature_matrix.pkl"))
snapshot_df = build_current_snapshot(df_clean)
print(f"Current snapshot: {len(snapshot_df)} customers scored as of {snapshot_df['obs_end'].iloc[0].date()}")
snapshot_df.to_pickle(os.path.join(PROCESSED_DIR, "current_snapshot.pkl"))

print("=== 4. Training XGBoost with chronological train/validation/test splits ===")
print(f"Embargo purge around each split boundary: {EMBARGO_DAYS} days")
X, y, groups, feature_cols = prepare_data(feature_df)
result = train_model(X, y, feature_cols, feature_df["obs_end"], groups)
model = result.model
print(f"Best trial PR-AUC (tuning half of validation, used for tuning only): {result.study.best_value:.4f}")
print(f"Final model trained on {result.final_fit_rows} rows (train split only)")
print(f"Validation period split by customer: {len(result.tune_idx)} tuning rows, {len(result.cal_idx)} calibration rows")

print("=== 5. Calibrating probabilities (on the calibration half the model and tuner never saw) ===")
calibration_func, calibrator = calibrate_probabilities(model, result.X_cal, result.y_cal)

print("=== 6. Selecting the profit threshold (on cross-fitted calibrated probabilities) ===")
cal_raw_probs = model.predict_proba(result.X_cal)[:, 1]
cal_oof_probs = cross_fitted_probabilities(cal_raw_probs, result.y_cal)
cal_spend = compute_avg_monthly_spend(
    feature_df.iloc[result.cal_idx]["monetary_total"].reset_index(drop=True)
)
optimal_threshold, threshold_results = find_optimal_threshold(
    result.y_cal.values, cal_oof_probs, cal_spend
)
print(f"Locked profit threshold selected without using the final test set: {optimal_threshold}")

print("=== 7. Final evaluation (test set used only after all choices are locked) ===")
test_raw_probs = model.predict_proba(result.X_test)[:, 1]
test_calibrated_probs = calibration_func(test_raw_probs)
metrics = compute_metrics(result.y_test, test_raw_probs, test_calibrated_probs)
print(f"PR-AUC (raw scores): {metrics['pr_auc']:.4f}")
print(f"Brier Score (calibrated probabilities): {metrics['brier_score']:.4f}")

test_spend = compute_avg_monthly_spend(
    feature_df.iloc[result.test_idx]["monetary_total"].reset_index(drop=True)
)

print("=== 8. Baseline comparison on final test set ===")
y_test_values = result.y_test.values
strategies = {
    f"Random ({RANDOM_TARGET_FRACTION:.0%})": evaluate_random_baseline(
        y_test_values, test_calibrated_probs, test_spend, RANDOM_TARGET_FRACTION
    ),
    f"Default Threshold ({DEFAULT_THRESHOLD})": evaluate_threshold_strategy(
        y_test_values, test_calibrated_probs, test_spend, DEFAULT_THRESHOLD
    ),
    "Profit-Optimized Threshold": evaluate_threshold_strategy(
        y_test_values, test_calibrated_probs, test_spend, optimal_threshold
    ),
    "Expected-Value Rule": evaluate_expected_value_strategy(
        y_test_values, test_calibrated_probs, test_spend
    ),
}

comparison_df = pd.DataFrame({
    "Strategy": list(strategies.keys()),
    "Total Interventions": [r["total_interventions"] for r in strategies.values()],
    "True Positives": [r["true_positives"] for r in strategies.values()],
    "Wasted Spend (FP)": [r["false_positives"] for r in strategies.values()],
    "Net Campaign Profit": [round(float(r["net_profit"]), 2) for r in strategies.values()],
})

print("\n=== Profit Comparison on Final Test Set ===")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(PROCESSED_DIR, "profit_comparison.csv"), index=False)
threshold_results.to_csv(os.path.join(PROCESSED_DIR, "threshold_analysis.csv"), index=False)

print("=== 9. Saving artifacts ===")
n_purged = len(X) - (len(result.train_idx) + len(result.tune_idx) + len(result.cal_idx) + len(result.test_idx))
index_sets = [set(result.train_idx), set(result.tune_idx), set(result.cal_idx), set(result.test_idx)]
splits_disjoint = all(
    index_sets[i].isdisjoint(index_sets[j])
    for i in range(len(index_sets)) for j in range(i + 1, len(index_sets))
)
validation_idx = np.concatenate([result.tune_idx, result.cal_idx])
artifacts = {
    "xgb_model.pkl": model,
    "calibrator.pkl": calibrator,
    "feature_names.pkl": result.feature_cols,
    "calibration_method.pkl": CALIBRATION_METHOD,
    "optimal_threshold.pkl": optimal_threshold,
    "metrics.pkl": {"pr_auc": metrics["pr_auc"], "brier_score": metrics["brier_score"]},
    "split_metadata.pkl": {
        "method": "chronological_train_validation_test",
        "final_model_trained_on": "train_only" if (
            splits_disjoint and result.final_fit_rows == len(result.train_idx)
        ) else "unverified",
        "final_fit_rows": result.final_fit_rows,
        "calibration_and_threshold_selection_period": "calibration half of validation, cross-fitted",
        "train_rows": len(result.train_idx),
        "tuning_rows": len(result.tune_idx),
        "calibration_rows": len(result.cal_idx),
        "validation_rows": len(result.tune_idx) + len(result.cal_idx),
        "test_rows": len(result.test_idx),
        "splits_disjoint": splits_disjoint,
        "embargo_days": EMBARGO_DAYS,
        "purged_rows": n_purged,
        "validation_end": str(feature_df.iloc[validation_idx]["obs_end"].max()),
        "test_start": str(feature_df.iloc[result.test_idx]["obs_end"].min()),
        "xgboost_version": xgb.__version__,
    }
}
for filename, value in artifacts.items():
    with open(os.path.join(ARTIFACTS_DIR, filename), "wb") as f:
        pickle.dump(value, f)

print("\nPipeline complete. Run 'streamlit run app/app.py' for dashboard.")
