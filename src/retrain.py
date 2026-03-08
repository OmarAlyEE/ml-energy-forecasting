# src/retrain.py

import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------ Project Paths ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_BATCH_PATH = os.path.join(BASE_DIR, "data", "processed", "new_batch.csv")
PRODUCTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "model.pkl")
STAGING_DIR = os.path.join(BASE_DIR, "models", "staging")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PRODUCTION_MODEL_PATH), exist_ok=True)

# ------------------ Features & Target ------------------
features = ["Voltage", "Global_intensity"]
target = "Global_active_power"

# ------------------ Load New Batch ------------------
if not os.path.exists(NEW_BATCH_PATH):
    raise FileNotFoundError(f"New batch data not found at {NEW_BATCH_PATH}")

new_df = pd.read_csv(NEW_BATCH_PATH)

# ------------------ Schema Validation ------------------
expected_columns = features + [target]
missing = [c for c in expected_columns if c not in new_df.columns]
if missing:
    raise ValueError(f"Missing columns in new batch: {missing}")

for col in expected_columns:
    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
new_df = new_df.dropna(subset=expected_columns)

# ------------------ Train/Test Split ------------------
X_new = new_df[features]
y_new = new_df[target]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# ------------------ Train New Model ------------------
new_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
new_model.fit(X_train_new, y_train_new)

# ------------------ Evaluate ------------------
# Load current production model if exists
if os.path.exists(PRODUCTION_MODEL_PATH):
    prod_model = joblib.load(PRODUCTION_MODEL_PATH)
    y_pred_prod = prod_model.predict(X_test_new)
    rmse_prod = np.sqrt(mean_squared_error(y_test_new, y_pred_prod))
else:
    rmse_prod = float("inf")  # No production model yet

# Evaluate new model
y_pred_new = new_model.predict(X_test_new)
rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

print(f"Current production RMSE: {rmse_prod:.4f}")
print(f"New model RMSE: {rmse_new:.4f}")

# ------------------ Log Metrics ------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_log_path = os.path.join(LOGS_DIR, f"retrain_log_{timestamp}.csv")
pd.DataFrame([{
    "timestamp": timestamp,
    "rmse_new": rmse_new,
    "mae_new": mae_new,
    "r2_new": r2_new,
    "rmse_prod": rmse_prod
}]).to_csv(metrics_log_path, index=False)
print(f"Metrics logged to {metrics_log_path}")

# ------------------ Save Model ------------------
# Save to staging
staging_path = os.path.join(STAGING_DIR, f"model_{timestamp}.pkl")
joblib.dump(new_model, staging_path)
print(f"New model saved to staging: {staging_path}")

# Promote to production if better
if rmse_new < rmse_prod:
    joblib.dump(new_model, PRODUCTION_MODEL_PATH)
    print(f"New model promoted to production: {PRODUCTION_MODEL_PATH}")
else:
    print("New model did not improve performance; kept in staging.")
