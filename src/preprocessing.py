"""
preprocessing.py
Loads the raw Telco Customer Churn dataset and produces a clean,
model-ready dataframe.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "data/Telco-Customer-Churn.csv"
TARGET = "Churn"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges has blank strings for customers with tenure 0; coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop the ID column, it carries no signal
    df = df.drop(columns=["customerID"])

    # Standardize "No internet service" / "No phone service" to "No"
    # to reduce redundant categories
    replace_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
    ]
    for col in replace_cols:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # Encode target as binary
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    # SeniorCitizen is already 0/1, keep as is but cast to category-friendly int
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light feature engineering that tends to help tree-based models."""
    df = df.copy()
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    df["NumServices"] = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["MultipleLines"] == "Yes").astype(int)
        + (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    return df


def get_clean_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df = clean(df)
    df = add_features(df)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = get_clean_data()
    print(df.shape)
    print(df.head())
    print(df[TARGET].value_counts(normalize=True))
