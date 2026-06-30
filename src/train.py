"""
train.py
Builds preprocessing + model pipelines for several algorithms,
trains them, and saves the best one to disk along with metrics.
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from preprocessing import get_clean_data, split_data, TARGET

MODEL_DIR = "models"
import os
os.makedirs(MODEL_DIR, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
        ]
    )
    return preprocessor


def get_candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(7043 * 0.73463) / (7043 * 0.26537),  # imbalance ratio
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    df = get_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    candidates = get_candidate_models()

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name}: {metrics}")

    # Pick best model by ROC AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} -> {results[best_name]}")

    joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.pkl")
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    # Save confusion matrix and classification report for the best model
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    with open(f"{MODEL_DIR}/best_model_report.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm}, f, indent=2)

    return results, best_name


if __name__ == "__main__":
    main()
