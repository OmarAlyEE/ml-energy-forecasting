# src/monitoring.py

import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.csv")

def log_prediction(input_data: dict, prediction: float):
    """Append prediction and input features to CSV log"""
    df = pd.DataFrame([{
        **input_data,
        "prediction": prediction,
        "timestamp": datetime.now()
    }])
    
    if not os.path.exists(PREDICTIONS_LOG):
        df.to_csv(PREDICTIONS_LOG, index=False)
    else:
        df.to_csv(PREDICTIONS_LOG, index=False, mode='a', header=False)
