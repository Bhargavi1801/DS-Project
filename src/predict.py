"""
predict.py
Loads the saved best model pipeline and scores new customer records.

Usage:
    python predict.py --input new_customers.csv --output predictions.csv
"""

import argparse
import joblib
import pandas as pd

from preprocessing import clean, add_features

MODEL_PATH = "models/best_model.pkl"


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning/feature steps used at training time.
    Expects raw-format input (same columns as the original Telco csv,
    Churn column optional/ignored)."""
    df = df.copy()
    if "Churn" not in df.columns:
        df["Churn"] = "No"  # placeholder so clean() doesn't break
    df = clean(df)
    df = add_features(df)
    return df.drop(columns=["Churn"])


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = pred
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw-format CSV of customers to score")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    model = load_model()
    raw = pd.read_csv(args.input)
    prepared = prepare_input(raw)
    result = predict(model, prepared)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
