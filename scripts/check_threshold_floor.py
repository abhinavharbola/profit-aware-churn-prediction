import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, ARTIFACTS_DIR
from src.modeling.trainer import prepare_data, temporal_train_val_test_split
from src.evaluation.profit_optimizer import find_optimal_threshold, compute_avg_monthly_spend

feature_matrix_path = os.path.join(PROCESSED_DIR, "feature_matrix.pkl")
model_path = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator.pkl")
calibration_method_path = os.path.join(ARTIFACTS_DIR, "calibration_method.pkl")

for path in [feature_matrix_path, model_path, calibrator_path, calibration_method_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/run_pipeline.py first.")

feature_df = pd.read_pickle(feature_matrix_path)
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(calibrator_path, "rb") as f:
    calibrator = pickle.load(f)
with open(calibration_method_path, "rb") as f:
    calibration_method = pickle.load(f)

X, y, groups, feature_cols = prepare_data(feature_df)
_, X_val, _, _, y_val, _, _, val_idx, _ = temporal_train_val_test_split(
    X, y, groups, feature_df["obs_end"]
)

raw_probs = model.predict_proba(X_val)[:, 1]
if calibration_method == "isotonic":
    calibrated_probs = calibrator.transform(raw_probs)
else:
    calibrated_probs = calibrator.predict_proba(np.array(raw_probs).reshape(-1, 1))[:, 1]

avg_monthly_spend = compute_avg_monthly_spend(
    feature_df.iloc[val_idx]["monetary_total"].reset_index(drop=True)
)
fine_thresholds = np.array([
    0.0001, 0.0005, 0.001, 0.002, 0.005,
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90
])
pipeline_thresholds = np.arange(0.01, 0.91, 0.01)
saved_threshold_path = os.path.join(ARTIFACTS_DIR, "optimal_threshold.pkl")
if not os.path.exists(saved_threshold_path):
    raise FileNotFoundError(f"{saved_threshold_path} not found. Run scripts/run_pipeline.py first.")
with open(saved_threshold_path, "rb") as f:
    saved_threshold = pickle.load(f)

reproduced_threshold, pipeline_results = find_optimal_threshold(
    y_val.values, calibrated_probs, avg_monthly_spend, thresholds=pipeline_thresholds
)

print(f"Reproduced pipeline threshold on pre-test validation data: {reproduced_threshold}")
print(f"Saved pipeline threshold: {saved_threshold}")
if float(reproduced_threshold) != float(saved_threshold):
    raise AssertionError(
        f"Saved threshold {saved_threshold} does not match reproduced threshold {reproduced_threshold}"
    )

print()
print("Pipeline threshold reproduction matches the saved artifact.")

fine_thresholds = np.array([
    0.0001, 0.0005, 0.001, 0.002, 0.005,
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90
])
fine_threshold, fine_results = find_optimal_threshold(
    y_val.values, calibrated_probs, avg_monthly_spend, thresholds=fine_thresholds
)
fine_results.insert(0, "requested_threshold", fine_thresholds)

print(f"Diagnostic finer-grid threshold (not used by the pipeline): {fine_threshold}")
print("This finer grid is diagnostic only and does not change the locked artifact.")
print(fine_results.to_string(index=False))
