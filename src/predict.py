# src/predict.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from src.monitoring import log_prediction  # import the logging function

# Initialize FastAPI app
app = FastAPI(title="Energy Forecasting API")

# -------------------------
# 1. Load the production model
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "model.pkl")

if not os.path.exists(PRODUCTION_MODEL_PATH):
    raise FileNotFoundError(f"Production model not found at {PRODUCTION_MODEL_PATH}")

model = joblib.load(PRODUCTION_MODEL_PATH)

# -------------------------
# 2. Define input data model
# -------------------------
class InputData(BaseModel):
    Voltage: float
    Global_intensity: float

# -------------------------
# 3. Define /predict endpoint
# -------------------------
@app.post("/predict")
def predict(input_data: InputData):
    input_dict = input_data.model_dump()
    df = pd.DataFrame([input_dict])
    
    # Make prediction
    prediction = float(model.predict(df)[0])
    
    # Log prediction for monitoring
    log_prediction(input_dict, prediction)
    
    return {"prediction": prediction}
