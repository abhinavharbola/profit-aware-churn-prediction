import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DIR, ARTIFACTS_DIR, DEFAULT_THRESHOLD,
    RANDOM_TARGET_FRACTION, CALIBRATION_METHOD
)
from src.data.cleaner import run_cleaning
from src.data.temporal import generate_windows
from src.features.rfm_engineer import build_feature_matrix
from src.modeling.trainer import prepare_data, train_model
from src.modeling.calibrator import calibrate_probabilities
from src.evaluation.metrics import compute_metrics
from src.evaluation.profit_optimizer import (
    find_optimal_threshold,
    evaluate_random_baseline,
    evaluate_default_baseline,
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

print("=== 4. Training XGBoost with chronological train/validation/test splits ===")
X, y, groups, feature_cols = prepare_data(feature_df)
model, study, feature_cols, X_val, y_val, X_test, y_test, val_idx, test_idx = train_model(
    X, y, groups, feature_cols, feature_df["obs_end"]
)
print(f"Best trial PR-AUC (validation set, used for tuning only): {study.best_value:.4f}")

print("=== 5. Calibrating probabilities (on the pre-test validation period) ===")
calibration_func, calibrator = calibrate_probabilities(model, X_val, y_val)

print("=== 6. Selecting the profit threshold (on the pre-test validation period) ===")
val_raw_probs = model.predict_proba(X_val)[:, 1]
val_calibrated_probs = calibration_func(val_raw_probs)
val_spend = compute_avg_monthly_spend(
    feature_df.iloc[val_idx]["monetary_total"].reset_index(drop=True)
)
optimal_threshold, threshold_results = find_optimal_threshold(
    y_val.values, val_calibrated_probs, val_spend
)
print(f"Locked profit threshold selected without using the final test set: {optimal_threshold}")

print("=== 7. Final evaluation (test set used only after all choices are locked) ===")
test_raw_probs = model.predict_proba(X_test)[:, 1]
test_calibrated_probs = calibration_func(test_raw_probs)
metrics = compute_metrics(y_test, test_calibrated_probs)
print(f"PR-AUC: {metrics['pr_auc']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")

test_spend = compute_avg_monthly_spend(
    feature_df.iloc[test_idx]["monetary_total"].reset_index(drop=True)
)

print("=== 8. Baseline comparison on final test set ===")
random_result = evaluate_random_baseline(
    y_test.values, test_calibrated_probs, test_spend, RANDOM_TARGET_FRACTION
)
default_result = evaluate_default_baseline(
    y_test.values, test_calibrated_probs, test_spend, DEFAULT_THRESHOLD
)
locked_result = evaluate_default_baseline(
    y_test.values, test_calibrated_probs, test_spend, optimal_threshold
)

comparison_df = pd.DataFrame({
    "Strategy": ["Random (20%)", "Default Threshold (0.5)", "Profit-Optimized"],
    "Total Interventions": [
        random_result["total_interventions"],
        default_result["total_interventions"],
        locked_result["total_interventions"]
    ],
    "True Positives": [
        random_result["true_positives"],
        default_result["true_positives"],
        locked_result["true_positives"]
    ],
    "Wasted Spend (FP)": [
        random_result["false_positives"],
        default_result["false_positives"],
        locked_result["false_positives"]
    ],
    "Net Campaign Profit": [
        f"£{random_result['net_profit']:,.0f}",
        f"£{default_result['net_profit']:,.0f}",
        f"£{locked_result['net_profit']:,.0f}"
    ]
})

print("\n=== Profit Comparison on Final Test Set ===")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(PROCESSED_DIR, "profit_comparison.csv"), index=False)
threshold_results.to_csv(os.path.join(PROCESSED_DIR, "threshold_analysis.csv"), index=False)

print("=== 9. Saving artifacts ===")
artifacts = {
    "xgb_model.pkl": model,
    "calibrator.pkl": calibrator,
    "feature_names.pkl": feature_cols,
    "calibration_method.pkl": CALIBRATION_METHOD,
    "optimal_threshold.pkl": optimal_threshold,
    "metrics.pkl": {"pr_auc": metrics["pr_auc"], "brier_score": metrics["brier_score"]},
    "split_metadata.pkl": {
        "method": "chronological_train_validation_test",
        "calibration_and_threshold_selection_period": "validation",
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "validation_end": str(feature_df.iloc[val_idx]["obs_end"].max()),
        "test_start": str(feature_df.iloc[test_idx]["obs_end"].min())
    }
}
for filename, value in artifacts.items():
    with open(os.path.join(ARTIFACTS_DIR, filename), "wb") as f:
        pickle.dump(value, f)

print("\nPipeline complete. Run 'streamlit run app/app.py' for dashboard.")
