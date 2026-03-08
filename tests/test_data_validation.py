import pandas as pd
from src.data_validation import validate_data

def test_schema():
    df = pd.DataFrame({
        "Voltage": [234.0, 235.0],
        "Global_intensity": [18.0, 20.0],
        "Global_active_power": [4.2, 4.5]
    })
    validate_data(df)
