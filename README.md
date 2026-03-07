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
