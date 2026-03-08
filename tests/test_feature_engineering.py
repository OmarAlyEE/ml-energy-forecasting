import pandas as pd
from src.train import features  # or import your preprocessing functions

def test_feature_columns_exist():
    df = pd.DataFrame({
        "Voltage": [230, 235],
        "Global_intensity": [15, 18]
    })
    for col in features:
        assert col in df.columns, f"Missing feature: {col}"

def test_numeric_conversion():
    df = pd.DataFrame({
        "Voltage": ["230", "235"],
        "Global_intensity": ["15", "18"]
    })
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    assert df[features].dtypes["Voltage"] == "int64" or df[features].dtypes["Voltage"] == "float64"
