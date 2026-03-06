
# src/data_validation.py
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Example schema for hourly energy data
schema = pa.DataFrameSchema({
    "timestamp": pa.Column(str),
    "energy_demand": pa.Column(float),
    "temperature": pa.Column(float),
    "humidity": pa.Column(int),
})

def validate_data(df: pd.DataFrame):
    schema.validate(df, lazy=True)  # lazy=True collects all errors
    print("✅ Data validation passed!")
