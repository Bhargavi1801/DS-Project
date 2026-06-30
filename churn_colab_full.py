"""
churn_colab_full.py
================================================================
CUSTOMER CHURN PREDICTION — FULL PROJECT (single-file Colab version)
Paste this whole file into one Google Colab cell and run it.
It downloads the dataset itself, so no uploads are required.
================================================================
"""

# !pip install -q xgboost   # uncomment if running outside Colab where xgboost isn't installed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
raw = pd.read_csv(url)
print("Raw shape:", raw.shape)
print(raw.head())

TARGET = "Churn"

# ------------------------------------------------------------
# 2. CLEANING & FEATURE ENGINEERING
# ------------------------------------------------------------
def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])

    replace_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]
    for col in replace_cols:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df

def add_features(df):
    df = df.copy()
    df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["TenureBucket"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                 labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
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

df = add_features(clean(raw))
print("\nClean shape:", df.shape)
print(df[TARGET].value_counts(normalize=True))
print(df.head())

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
df[TARGET].map({0: "No", 1: "Yes"}).value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("Churn distribution"); ax.set_xlabel("Churn"); ax.set_ylabel("Customers")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.kdeplot(data=df, x=col, hue=df[TARGET].map({0: "No", 1: "Yes"}), ax=ax, fill=True, common_norm=False)
    ax.set_title(f"{col} by churn")
plt.tight_layout(); plt.show()

cat_cols_eda = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "SeniorCitizen"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for ax, col in zip(axes, cat_cols_eda):
    df.groupby(col)[TARGET].mean().sort_values(ascending=False).plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_title(f"Churn rate by {col}"); ax.set_ylabel("Churn rate"); ax.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 4. TRAIN / TEST SPLIT + PREPROCESSING PIPELINE
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
])

print("\nTrain/test shapes:", X_train.shape, X_test.shape)

# ------------------------------------------------------------
# 5. TRAIN & COMPARE MODELS
# ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                              random_state=42, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
}

results, pipelines = {}, {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    pipelines[name] = pipe

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print("\nModel comparison:")
print(results_df)

# ------------------------------------------------------------
# 6. BEST MODEL EVALUATION
# ------------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

best_name = results_df.index[0]
best_pipeline = pipelines[best_name]
print("\nBest model:", best_name)

y_pred = best_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix — {best_name}")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
for name, pipe in pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_title("ROC curves — model comparison")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ------------------------------------------------------------
model_step = best_pipeline.named_steps["model"]
preproc_step = best_pipeline.named_steps["preprocessor"]
feature_names = preproc_step.get_feature_names_out()

importances = model_step.feature_importances_ if hasattr(model_step, "feature_importances_") \
    else np.abs(model_step.coef_[0])

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="#4C72B0")
ax.set_title(f"Top 15 features — {best_name}")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 8. SAVE MODEL
# ------------------------------------------------------------
import joblib
joblib.dump(best_pipeline, "best_model.pkl")
print("\nSaved best_model.pkl —", best_name)

# Uncomment to download in Colab:
# from google.colab import files
# files.download("best_model.pkl")

# ------------------------------------------------------------
# 9. PREDICT ON NEW DATA
# ------------------------------------------------------------
def predict_new(raw_df, pipeline=best_pipeline):
    df_new = raw_df.copy()
    if "Churn" not in df_new.columns:
        df_new["Churn"] = "No"
    df_new = add_features(clean(df_new))
    df_new = df_new.drop(columns=["Churn"])
    proba = pipeline.predict_proba(df_new)[:, 1]
    df_new["churn_probability"] = proba
    df_new["churn_prediction"] = (proba >= 0.5).astype(int)
    return df_new

# Example: score the first 10 rows of the original raw data
sample_preds = predict_new(raw.head(10))
print("\nSample predictions:")
print(sample_preds[["churn_probability", "churn_prediction"]])
