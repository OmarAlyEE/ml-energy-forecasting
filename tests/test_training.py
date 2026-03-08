import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.train import features, target

def test_model_training_output():
    df = pd.DataFrame({
        "Voltage": [230, 235, 240],
        "Global_intensity": [15, 18, 20],
        "Global_active_power": [3.5, 4.2, 4.5]
    })
    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(y), "Prediction length mismatch"
    assert all(isinstance(p, float) for p in pred), "Predictions are not floats"
