"""
SHAP Interpretability Script (Stable Version)
Supports both XGBClassifier (sklearn API) and Booster models.
"""

import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

from model_validation_project import load_data, basic_preprocess
import joblib
import os


warnings.filterwarnings("ignore")

# === CONFIG ===
DATA_PATH = "Steven_Maxwell_Portfolio/projects/model_validation_practice/model_validation_project/data/application_train.csv"
TARGET_COL = "TARGET"
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR = "Steven_Maxwell_Portfolio/projects/model_validation_practice/model_validation_project/outputs/shap_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === LOAD DATA ===
df = pd.read_csv(DATA_PATH, nrows=5000)

# Prepare features and target
df_num = basic_preprocess(df)
X = df_num.drop(columns=[TARGET_COL])
y = df_num['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# === TRAIN MODEL ===
print("Training XGBoost model...")

try:
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Trained using sklearn XGBClassifier.")
    model_type = "sklearn"
except Exception as e:
    print("Sklearn API failed, switching to Booster:", e)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_STATE
    }
    model = xgb.train(params, dtrain, num_boost_round=200)
    model_type = "booster"
    print("Trained using raw Booster.")

# === SHAP EXPLAINER ===
print("Initializing SHAP Explainer...")

if model_type == "sklearn":
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
else:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)

print("SHAP values computed successfully.")

# === MODEL PERFORMANCE ===
try:
    preds = model.predict_proba(X_test)[:, 1]
except AttributeError:
    preds = model.predict(xgb.DMatrix(X_test))

auc = roc_auc_score(y_test, preds)
print(f"Model AUC: {auc:.4f}")

# === VISUALIZATION ===
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, show=False)
print("SHAP summary plot generated successfully.")

# Save SHAP values and supporting data
joblib.dump(shap_values, os.path.join(OUTPUT_DIR, "shap_values.pkl"))
joblib.dump(X_test, os.path.join(OUTPUT_DIR, "X_test.pkl"))
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))

print("SHAP values, test data, and model saved successfully.")