# src/train.py

import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json

# ---------------------------------------------------
# 1. Project Paths (robust absolute paths)
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "household_power_consumption.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STAGING_DIR = os.path.join(MODELS_DIR, "staging")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("Project directory:", BASE_DIR)
print("Dataset path:", DATA_PATH)

# ---------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"],
    low_memory=False
)

print("Raw dataset shape:", df.shape)

# ---------------------------------------------------
# 3. Feature Engineering / Preprocessing
# ---------------------------------------------------

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
df = df.drop(columns=["Date", "Time"])

target = "Global_active_power"
df = df.dropna(subset=[target])

features = ["Voltage", "Global_intensity"]

# Convert features to numeric & fill missing
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df[target]

print("\nSample after preprocessing:")
print(df.head())

# ---------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Save test set for evaluate.py
test_df = X_test.copy()
test_df[target] = y_test
test_csv_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
test_df.to_csv(test_csv_path, index=False)
print(f"Test set saved to {test_csv_path}")

# ---------------------------------------------------
# 5. Train Model
# ---------------------------------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining model...")
model.fit(X_train, y_train)

# ---------------------------------------------------
# 6. Evaluate Model (optional logging)
# ---------------------------------------------------

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Save metrics log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log = {
    "timestamp": timestamp,
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2)
}
log_path = os.path.join(LOGS_DIR, f"train_log_{timestamp}.json")
with open(log_path, "w") as f:
    json.dump(log, f, indent=4)
print(f"Training log saved to {log_path}")

# ---------------------------------------------------
# 7. Save Model to staging/
# ---------------------------------------------------

staging_model_path = os.path.join(STAGING_DIR, f"model_{timestamp}.pkl")
joblib.dump(model, staging_model_path)
print(f"Model saved to staging at {staging_model_path}")
