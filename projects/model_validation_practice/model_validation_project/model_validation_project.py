import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

# -------------------------
# Utilities
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f'Loaded {df.shape[0]} rows and {df.shape[1]} columns')
    return df

def basic_preprocess(df):
    # Keep numeric columns for a quick baseline validation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['SK_ID_CURR','TARGET']]
    df_num = df[numeric_cols + ['TARGET']].copy()
    # Simple median imputation for numeric columns
    df_num = df_num.fillna(df_num.median())
    return df_num

def train_baseline_models(X_train, y_train, X_val, y_val):
    # Logistic Regression baseline
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X_train, y_train)
    y_val_lr = lr.predict_proba(X_val)[:,1]
    auc_lr = roc_auc_score(y_val, y_val_lr)
    print('Logistic Regression AUC:', auc_lr)

    # XGBoost baseline
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {'objective':'binary:logistic','eval_metric':'auc','learning_rate':0.1,'max_depth':6,'seed':42}
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval,'val')], early_stopping_rounds=10, verbose_eval=False)
    bst.save_model("Steven_Maxwell_Portfolio/projects/model_validation_practice/model_validation_project/models/xgb_credit_model.json")
    y_val_xgb = bst.predict(dval)
    auc_xgb = roc_auc_score(y_val, y_val_xgb)
    print('XGBoost AUC:', auc_xgb)

    return lr, bst, y_val_lr, y_val_xgb

def psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI) - simple implementation
    expected, actual: 1D arrays of predicted probabilities (or any continuous variable)
    """
    expected_perc, _ = np.histogram(expected, bins=buckets)[0], None
    actual_perc, _ = np.histogram(actual, bins=buckets)[0], None
    expected_perc = expected_perc / np.sum(expected_perc)
    actual_perc = actual_perc / np.sum(actual_perc)
    # avoid zeros
    expected_perc = np.where(expected_perc==0, 1e-8, expected_perc)
    actual_perc = np.where(actual_perc==0, 1e-8, actual_perc)
    return np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))

def stress_test_feature(bst, X_test, y_test, feature, multiplier):
    """Simple stress test: multiply a numeric feature by multiplier and compute AUC"""
    if feature not in X_test.columns:
        print(f'Feature {feature} not found - skipping stress test')
        return None
    X_stress = X_test.copy()
    X_stress[feature] = X_stress[feature] * multiplier
    auc = roc_auc_score(y_test, bst.predict(xgb.DMatrix(X_stress)))
    print(f'AUC under stress (feature {feature} * {multiplier}): {auc:.4f}')
    return auc

# -------------------------
# Main flow
# -------------------------
def main():
    DATA_DIR = 'Steven_Maxwell_Portfolio/projects/model_validation_practice/model_validation_project/data'
    TRAIN_FILE = os.path.join(DATA_DIR, 'application_train.csv')
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f'Place application_train.csv in {TRAIN_FILE} (download from Kaggle)')

    # Load and preprocess
    df = load_data(TRAIN_FILE)
    df_num = basic_preprocess(df)

    # Train / Val / Test split (stratified)
    X = df_num.drop(columns=['TARGET'])
    y = df_num['TARGET']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print('Shapes -> ', X_train.shape, X_val.shape, X_test.shape)

    # Train baseline models
    lr, bst, y_val_lr, y_val_xgb = train_baseline_models(X_train, y_train, X_val, y_val)

    # Test set performance
    auc_test_xgb = roc_auc_score(y_test, bst.predict(xgb.DMatrix(X_test)))
    print('Test AUC (XGBoost):', auc_test_xgb)

    # PSI check between train and validation predicted probabilities
    # PSI value < 0.1 equates to a stable model
    train_pred_lr = lr.predict_proba(X_train)[:,1]
    val_pred_lr = lr.predict_proba(X_val)[:,1]
    psi_val = psi(train_pred_lr, val_pred_lr)
    print('PSI (train -> val):', psi_val)

    # Stress test example: reduce income by 20%
    stress_test_feature(bst, X_test, y_test, 'AMT_INCOME_TOTAL', 0.8)

    summary = {
        'lr_val_auc': float(roc_auc_score(y_val, y_val_lr)),
        'xgb_val_auc': float(roc_auc_score(y_val, y_val_xgb)),
        'xgb_test_auc': float(auc_test_xgb),
        'psi_train_val': float(psi_val)
    }
    print('Validation summary:', summary)

if __name__ == '__main__':
    main()
