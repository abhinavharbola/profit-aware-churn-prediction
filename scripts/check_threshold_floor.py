import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, ARTIFACTS_DIR, EMBARGO_DAYS
from src.modeling.trainer import prepare_data, temporal_train_val_test_split, split_validation_by_customer
from src.modeling.calibrator import cross_fitted_probabilities
from src.evaluation.profit_optimizer import find_optimal_threshold, compute_avg_monthly_spend, THRESHOLD_GRID

feature_matrix_path = os.path.join(PROCESSED_DIR, "feature_matrix.pkl")
model_path = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
calibration_method_path = os.path.join(ARTIFACTS_DIR, "calibration_method.pkl")
split_metadata_path = os.path.join(ARTIFACTS_DIR, "split_metadata.pkl")
saved_threshold_path = os.path.join(ARTIFACTS_DIR, "optimal_threshold.pkl")

for path in [feature_matrix_path, model_path, calibration_method_path, split_metadata_path, saved_threshold_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/run_pipeline.py first.")

feature_df = pd.read_pickle(feature_matrix_path)
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(calibration_method_path, "rb") as f:
    calibration_method = pickle.load(f)
with open(split_metadata_path, "rb") as f:
    split_metadata = pickle.load(f)
with open(saved_threshold_path, "rb") as f:
    saved_threshold = pickle.load(f)

saved_embargo_days = split_metadata.get("embargo_days")
if saved_embargo_days is not None and saved_embargo_days != EMBARGO_DAYS:
    raise AssertionError(
        f"split_metadata.pkl was produced with EMBARGO_DAYS={saved_embargo_days}, "
        f"but config.py currently has EMBARGO_DAYS={EMBARGO_DAYS}. Reproducing the "
        "split below with the current config would not match the split the saved "
        "artifacts were actually trained/calibrated on. Re-run scripts/run_pipeline.py "
        "or restore the original EMBARGO_DAYS before trusting this script's output."
    )

X, y, groups, feature_cols = prepare_data(feature_df)
_, X_val, _, _, y_val, _, train_idx, val_idx, _ = temporal_train_val_test_split(
    X, y, feature_df["obs_end"], embargo_days=EMBARGO_DAYS
)
_, cal_pos = split_validation_by_customer(val_idx, groups)
X_cal = X_val.iloc[cal_pos].reset_index(drop=True)
y_cal = y_val.iloc[cal_pos].reset_index(drop=True)
cal_idx = val_idx[cal_pos]

if split_metadata.get("final_fit_rows") != len(train_idx) or split_metadata.get("train_rows") != len(train_idx):
    raise AssertionError(
        "The saved model's recorded fit size does not equal the reproduced train split, "
        "so the model may have been fit on validation or test rows. Re-run "
        "scripts/run_pipeline.py with the current trainer before trusting this script's output."
    )
if len(cal_idx) != split_metadata.get("calibration_rows"):
    raise AssertionError(
        "Reproduced calibration split size does not match the split recorded at training "
        "time, the saved artifacts were likely produced by a different feature matrix or "
        "config than the one currently on disk."
    )

raw_probs = model.predict_proba(X_cal)[:, 1]
calibrated_probs = cross_fitted_probabilities(raw_probs, y_cal, method=calibration_method)

avg_monthly_spend = compute_avg_monthly_spend(
    feature_df.iloc[cal_idx]["monetary_total"].reset_index(drop=True)
)
pipeline_thresholds = THRESHOLD_GRID

reproduced_threshold, pipeline_results = find_optimal_threshold(
    y_cal.values, calibrated_probs, avg_monthly_spend, thresholds=pipeline_thresholds
)

print(f"Reproduced pipeline threshold on the calibration half: {reproduced_threshold}")
print(f"Saved pipeline threshold: {saved_threshold}")
if float(reproduced_threshold) != float(saved_threshold):
    raise AssertionError(
        f"Saved threshold {saved_threshold} does not match reproduced threshold {reproduced_threshold}"
    )

print()
print("Pipeline threshold reproduction matches the saved artifact.")
print(
    "Note: this confirms the threshold sweep is deterministic given these inputs and that "
    "the recorded fit size equals the reproduced train split. It does not replace the "
    "leakage regression test in tests/test_no_calibration_leakage.py."
)

fine_thresholds = np.array([
    0.0001, 0.0005, 0.001, 0.002, 0.005,
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90
])
fine_threshold, fine_results = find_optimal_threshold(
    y_cal.values, calibrated_probs, avg_monthly_spend, thresholds=fine_thresholds, round_decimals=4
)
fine_results.insert(0, "requested_threshold", fine_thresholds)

print(f"Diagnostic finer-grid threshold (not used by the pipeline): {fine_threshold}")
print("This finer grid is diagnostic only and does not change the locked artifact.")
print(fine_results.to_string(index=False))
