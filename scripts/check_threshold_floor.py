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
split_metadata_path = os.path.join(ARTIFACTS_DIR, "split_metadata.pkl")

for path in [feature_matrix_path, model_path, calibrator_path, calibration_method_path, split_metadata_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/run_pipeline.py first.")

feature_df = pd.read_pickle(feature_matrix_path)
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(calibrator_path, "rb") as f:
    calibrator = pickle.load(f)
with open(calibration_method_path, "rb") as f:
    calibration_method = pickle.load(f)
with open(split_metadata_path, "rb") as f:
    split_metadata = pickle.load(f)

if split_metadata.get("final_model_trained_on") != "train_only":
    raise AssertionError(
        "split_metadata.pkl does not confirm the shipped model was trained on the "
        "train split alone. Everything below assumes calibration and thresholding "
        "run against data the model never trained on; re-run scripts/run_pipeline.py "
        "with the current trainer before trusting this script's output."
    )

X, y, groups, feature_cols = prepare_data(feature_df)
X_train, X_val, _, y_train, y_val, _, train_idx, val_idx, _ = temporal_train_val_test_split(
    X, y, feature_df["obs_end"]
)

if set(train_idx) & set(val_idx):
    raise AssertionError("Train and validation indices overlap, the split itself is broken.")
if len(X_val) != split_metadata.get("validation_rows"):
    raise AssertionError(
        "Reproduced validation split size does not match the split recorded at training "
        "time, the saved artifacts were likely produced by a different feature matrix or "
        "config than the one currently on disk."
    )

raw_probs = model.predict_proba(X_val)[:, 1]
if calibration_method == "isotonic":
    calibrated_probs = calibrator.transform(raw_probs)
else:
    calibrated_probs = calibrator.predict_proba(np.array(raw_probs).reshape(-1, 1))[:, 1]

avg_monthly_spend = compute_avg_monthly_spend(
    feature_df.iloc[val_idx]["monetary_total"].reset_index(drop=True)
)
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
print(
    "Note: this confirms the threshold sweep is deterministic given these inputs, and "
    "the split_metadata check above confirms the model that produced these inputs was "
    "trained on the train split alone. Neither check replaces a dedicated leakage "
    "regression test, see tests/test_no_calibration_leakage.py for that."
)

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
