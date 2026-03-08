# ML Energy Forecasting Pipeline

Fully automated Machine Learning pipeline for forecasting hourly energy consumption. Includes CI, model versioning, deployment, and monitoring.

---

## Problem Statement

Predict hourly energy consumption to help energy providers optimize load distribution and detect anomalies in real-time. The pipeline uses voltage and current intensity measurements as input features.

---

## Dataset

This project uses the [Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from UCI.

**Instructions:**  
1. Download the dataset from the above link.  
2. Place the downloaded file here: `data/raw/household_power_consumption.txt`.

---

## Model Performance

After training and evaluation using a Random Forest model:

| Metric        | Value  |
|---------------|--------|
| Baseline RMSE | 1.0602 |
| Model RMSE    | 0.0369 |
| MAE           | 0.0201 |
| R²            | 0.9983 |

Results were obtained on the test set (`data/processed/test.csv`).

---

## Architecture
Data (raw CSV)
↓
Data Validation (Pandera)
↓
Feature Engineering
↓
Training Pipeline (scikit-learn, RandomForest)
↓
Model Registry (staging/production)
↓
API Deployment (FastAPI + Docker)
↓
Monitoring & Drift Detection
↓
Retraining Trigger

## Tech Stack

- **Python 3.11**  
- **scikit-learn** (modeling)  
- **Pandera** (data validation)  
- **FastAPI** (API deployment)  
- **Docker** (containerization)  
- **GitHub Actions** (CI/CD)  
- **MLflow** (optional, experiment tracking)

---

## How to Run Locally

Clone the repository:

git clone https://github.com/<your-username>/ml-energy-forecasting.git
cd ml-energy-forecasting

Install dependencies:
pip install -r requirements.txt

Run tests:
python -m pytest tests/

Start API server:
python -m uvicorn src.predict:app --reload --host 127.0.0.1 --port 8000

Make predictions:
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"Voltage\":234.0,\"Global_intensity\":18.0}"


## CI/CD

GitHub Actions workflow validates the data schema and runs all unit tests on every push.
Any failure in tests or schema validation blocks merging to main.
Ensures professional-grade automation and safeguards against bad data or broken code.

## Retraining Workflow

New batch data triggers retrain.py.
Data is validated before training.
New model is evaluated against production.
If performance improves (lower RMSE), the model is promoted to production.
Logs are saved in logs/ for transparency.

## Future Improvements

Add MLflow experiment tracking for detailed model comparisons.

Incorporate automated feature selection or hyperparameter tuning.

Expand monitoring to detect concept drift using statistical tests (e.g., KS test).

Add time-series specific models (LSTM, XGBoost, Prophet) for better accuracy.

## Scaling Considerations

Docker container can be deployed on cloud services (AWS, GCP, Azure).

CI/CD workflow can integrate with cloud retraining triggers.

Monitoring can be extended to a dashboard for real-time alerts.

Model registry ensures safe rollback and version control.


OmarAlyEE © 2026
