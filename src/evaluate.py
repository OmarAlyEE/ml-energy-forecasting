
# src/evaluate.py
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# -------------------------
# 1. Define paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGING_DIR = os.path.join(BASE_DIR, "models", "staging")

# Get all staging models
staging_models = [f for f in os.listdir(STAGING_DIR) if f.endswith(".pkl")]
if not staging_models:
    raise FileNotFoundError(f"No staging models found in {STAGING_DIR}")

# Pick the latest by filename (timestamps in filename)
staging_models.sort()  # ascending order
STAGING_MODEL_PATH = os.path.join(STAGING_DIR, staging_models[-1])

print(f"Using latest staging model: {STAGING_MODEL_PATH}")
PRODUCTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "model.pkl")
ARCHIVE_DIR = os.path.join(BASE_DIR, "models", "archive")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")  

# Ensure directories exist
os.makedirs(os.path.dirname(STAGING_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PRODUCTION_MODEL_PATH), exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# -------------------------
# 2. Load test data
# -------------------------
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

df = pd.read_csv(TEST_DATA_PATH)
features = ["Voltage", "Global_intensity"]  
target = "Global_active_power"

X_test = df[features]
y_test = df[target]

# -------------------------
# 3. Load models
# -------------------------
if not os.path.exists(STAGING_MODEL_PATH):
    raise FileNotFoundError(f"No staging model found at {STAGING_MODEL_PATH}")

staging_model = joblib.load(STAGING_MODEL_PATH)

production_model = None
if os.path.exists(PRODUCTION_MODEL_PATH):
    production_model = joblib.load(PRODUCTION_MODEL_PATH)

# -------------------------
# 4. Evaluate staging model
# -------------------------
y_pred = staging_model.predict(X_test)

staging_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
staging_mae = mean_absolute_error(y_test, y_pred)
staging_r2 = r2_score(y_test, y_pred)

print(f"Staging model performance -- RMSE: {staging_rmse:.4f}, MAE: {staging_mae:.4f}, R2: {staging_r2:.4f}")

# -------------------------
# 5. Compare with production
# -------------------------
promote = False
prod_rmse = None

if production_model is not None:
    y_prod_pred = production_model.predict(X_test)
    prod_rmse = np.sqrt(mean_squared_error(y_test, y_prod_pred))
    print(f"Production model RMSE: {prod_rmse:.4f}")
    
    if staging_rmse < prod_rmse:  # improvement
        promote = True
else:
    # No production model yet → automatically promote staging
    promote = True
    print("No production model found. Staging will be promoted.")

# -------------------------
# 6. Promote or discard
# -------------------------
if promote:
    print("Promoting staging model to production...")

    # Archive old production model if exists
    if production_model is not None:
        archive_path = os.path.join(ARCHIVE_DIR, f"model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        joblib.dump(production_model, archive_path)
        print(f"Old production model archived to {archive_path}")

    # Promote staging to production
    joblib.dump(staging_model, PRODUCTION_MODEL_PATH)
    print(f"Staging model promoted to production at {PRODUCTION_MODEL_PATH}")
else:
    print("Staging model did not improve. Discarding.")

# -------------------------
# 7. Save evaluation log 
# -------------------------
log = {
    "staging_model": STAGING_MODEL_PATH,
    "staging_rmse": float(staging_rmse),
    "staging_mae": float(staging_mae),
    "staging_r2": float(staging_r2),
    "production_rmse": float(prod_rmse) if prod_rmse is not None else None,
    "promoted": promote
}

log_path = os.path.join(LOGS_DIR, f"evaluate_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(log_path, "w") as f:
    json.dump(log, f, indent=4)

print(f"Evaluation log saved to {log_path}")
