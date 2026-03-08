# src/drift_check.py

import os
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
PREDICTIONS_LOG = os.path.join(BASE_DIR, "logs", "predictions.csv")

# Drift thresholds
KS_THRESHOLD = 0.05
MEAN_SHIFT_THRESHOLD = 0.1  # 10% change in mean

def check_drift():
    if not os.path.exists(PREDICTIONS_LOG):
        print("No new predictions to check for drift.")
        return False

    # Load training features
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    new_df = pd.read_csv(PREDICTIONS_LOG)

    drift_detected = False

    features = ["Voltage", "Global_intensity"]
    for f in features:
        # KS test
        ks_stat, ks_p = ks_2samp(train_df[f], new_df[f])
        print(f"{f} -- KS p-value: {ks_p:.4f}")
        if ks_p < KS_THRESHOLD:
            print(f"⚠ Drift detected on {f} (KS test)")
            drift_detected = True

        # Mean shift detection
        mean_train = np.mean(train_df[f])
        mean_new = np.mean(new_df[f])
        shift = abs(mean_new - mean_train) / (mean_train + 1e-6)
        print(f"{f} -- Mean shift: {shift:.4f}")
        if shift > MEAN_SHIFT_THRESHOLD:
            print(f"⚠ Drift detected on {f} (Mean shift)")
            drift_detected = True

    if drift_detected:
        print("🚨 Data drift detected! Consider retraining.")
    else:
        print("✅ No significant drift detected.")

    return drift_detected

if __name__ == "__main__":
    check_drift()
