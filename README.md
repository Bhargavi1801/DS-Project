# Customer Churn Prediction — Telco Dataset

End-to-end churn prediction project: EDA, feature engineering, model comparison,
and a deployable scoring script.

## Project Structure

```
churn_project/
├── data/
│   └── Telco-Customer-Churn.csv     # raw dataset (IBM Telco Customer Churn, 7,043 rows)
├── notebooks/
│   └── churn_analysis.ipynb         # full EDA + modeling walkthrough, charts included
├── src/
│   ├── preprocessing.py             # cleaning + feature engineering
│   ├── train.py                     # trains & compares 4 models, saves the best
│   └── predict.py                   # CLI scorer for new customer batches
├── models/
│   ├── best_model.pkl                # saved best pipeline (preprocessing + model)
│   ├── metrics.json                  # metrics for every model trained
│   └── best_model_report.json        # classification report + confusion matrix
├── report/
│   └── Churn_Prediction_Report.docx  # written project report with figures
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Retrain from scratch:**
```bash
cd src
python train.py
```

**Score new customers** (CSV in the same raw column format as the original dataset, `Churn` column optional):
```bash
cd src
python predict.py --input new_customers.csv --output predictions.csv
```

**Explore the analysis:**
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

## Results Summary

| Model | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| **Logistic Regression (selected)** | 73.9% | 50.5% | 80.2% | **0.845** |
| Random Forest | 76.9% | 55.0% | 72.5% | 0.842 |
| Gradient Boosting | 80.0% | 66.0% | 50.8% | 0.842 |
| XGBoost | 74.9% | 51.8% | 77.5% | 0.837 |

Logistic Regression was selected for the best ROC AUC and highest recall — prioritizing
catching as many actual churners as possible, which fits a proactive retention use case.

## Key Churn Drivers

- Month-to-month contracts (vs. 1-2 year contracts)
- Low tenure (newer customers)
- Fiber optic internet service
- Electronic check payment method
- Missing TechSupport / OnlineSecurity add-ons
- Senior citizen status

See `report/Churn_Prediction_Report.docx` for the full write-up, charts, and business
recommendations.
