# tests/test_data_validation.py
import pandas as pd
from src.data_validation import validate_data

def test_schema():
    df = pd.DataFrame({
        "timestamp": ["2026-03-01 00:00", "2026-03-01 01:00"],
        "energy_demand": [100.5, 110.3],
        "temperature": [22.1, 21.8],
        "humidity": [45, 50],
    })
    validate_data(df)
