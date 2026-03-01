
# src/data_validation.py
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Example schema for hourly energy data
schema = DataFrameSchema({
    "timestamp": Column(pa.DateTime),
    "energy_demand": Column(pa.Float, Check.ge(0)),  # demand cannot be negative
    "temperature": Column(pa.Float),
    "humidity": Column(pa.Float),
})

def validate_data(df: pd.DataFrame):
    schema.validate(df, lazy=True)  # lazy=True collects all errors
    print("✅ Data validation passed!")
