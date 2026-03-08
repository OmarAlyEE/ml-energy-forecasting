# ml-energy-forecasting
Fully automated Machine Learning pipeline for forecasting energy demands. Incudes CI, model versioning, deployment, and monitoring.
CI test trigger
## Dataset

This project uses the [Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) from UCI.

**Instructions:**  
1. Download the dataset from the above link.  
2. Place the downloaded file here: `data/raw/household_power_consumption.txt`.
## Model Performance

After training and evaluation:

| Metric          | Value  |
|-----------------|--------|
| Baseline RMSE   | 1.0602 |
| Model RMSE      | 0.0369 |
| MAE             | 0.0201 |
| R²              | 0.9983 |



These results were obtained on the test set (`data/processed/test.csv`) using the Random Forest model.

Problem Statement:

Predict hourly energy consumption based on real-time voltage and current intensity measurements. This enables energy providers to optimize load distribution and detect anomalies proactively.

Architecture:
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

Tech Stack:

Python 3.11

scikit-learn (modeling)

Pandera (data validation)

FastAPI (API deployment)

Docker (containerization)

GitHub Actions (CI/CD)

MLflow (optional, experiment tracking)

How to Run Locally:

Clone the repo:

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
CI/CD:

GitHub Actions workflow validates the data schema and runs all unit tests on every push.

Any failure in tests or schema validation blocks merging to main.

Retraining Workflow:

New batch data triggers retrain.py.

Data is validated before training.

New model is evaluated against production.

If performance improves (RMSE lower), the model is promoted to production.

Logs are saved in logs/ for transparency.

Future Improvements:

Add MLflow experiment tracking for detailed model comparisons.

Incorporate automated feature selection or hyperparameter tuning.

Expand monitoring to detect concept drift with statistical tests (KS test).

Add time-series specific models (LSTM, XGBoost, Prophet) for better accuracy.

scaling Considerations:

Docker container can be deployed on cloud services (AWS, GCP, Azure).

CI/CD workflow can integrate with cloud retraining triggers.

Monitoring can be extended to a dashboard for real-time alerts.
