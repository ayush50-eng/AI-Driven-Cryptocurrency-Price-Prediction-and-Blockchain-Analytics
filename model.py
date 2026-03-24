"""
model.py — Credit Card Fraud Detection ML Pipeline
====================================================
Steps:
  1. Load dataset from CSV
  2. Preprocess (normalize Amount & Time)
  3. Handle class imbalance via SMOTE
  4. Train Logistic Regression + Random Forest
  5. Evaluate both models
  6. Save best model as model.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "creditcard.csv")
MODEL_OUTPUT  = "model.pkl"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load the Kaggle credit card fraud CSV."""
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at '{path}'.")
        print("  -> Download creditcard.csv from Kaggle and place it in the 'dataset/' folder.")
        sys.exit(1)

    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape : {df.shape}")
    print(f"[INFO] Class distribution:\n{df['Class'].value_counts().to_string()}\n")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Normalize 'Time' and 'Amount' using StandardScaler.
    The V1-V28 features are already PCA-transformed (zero-mean).
    """
    print("[INFO] Preprocessing: normalizing 'Time' and 'Amount' ...")

    scaler = StandardScaler()

    df = df.copy()
    df["scaled_time"]   = scaler.fit_transform(df[["Time"]])
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])

    # Drop original columns; the scaler used for inference will be rebuilt below
    df.drop(columns=["Time", "Amount"], inplace=True)

    # Features matrix X and label vector y
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Rebuild a combined scaler on the full feature set for later use
    feature_scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        feature_scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y, feature_scaler


# ─────────────────────────────────────────────
# 3. HANDLE CLASS IMBALANCE — SMOTE
# ─────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """Over-sample the minority (fraud) class with SMOTE."""
    print("[INFO] Applying SMOTE to balance classes ...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE — shape: {X_res.shape} | "
          f"Class counts: {dict(zip(*np.unique(y_res, return_counts=True)))}\n")
    return X_res, y_res


# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("[INFO] Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_train, y_train)
    print("[INFO] Logistic Regression trained.\n")
    return lr


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("[INFO] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("[INFO] Random Forest trained.\n")
    return rf


# ─────────────────────────────────────────────
# 5. EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Print and return evaluation metrics."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("=" * 50)
    print(f"  Model : {name}")
    print("-" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
    print("=" * 50 + "\n")

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model, scaler, path: str):
    """Persist model AND its feature scaler together as a bundle."""
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, path)
    print(f"[INFO] Model bundle saved -> {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # 1. Load
    df = load_data(DATASET_PATH)

    # 2. Preprocess
    X, y, scaler = preprocess(df)

    # 3. Train/Test split BEFORE SMOTE (important — never leak test data into SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

    # 4. SMOTE on training set only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 5. Train both models
    lr_model = train_logistic_regression(X_train_res, y_train_res)
    rf_model = train_random_forest(X_train_res, y_train_res)

    # 6. Evaluate both
    print("\n[RESULTS] Evaluation on held-out test set:\n")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # 7. Pick best model by F1-score
    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best_model = rf_model
        winner = "Random Forest"
    else:
        best_model = lr_model
        winner = "Logistic Regression"

    print(f"[INFO] Best model selected: {winner} (F1={max(rf_metrics['f1'], lr_metrics['f1']):.4f})\n")

    # 8. Save
    save_model(best_model, scaler, MODEL_OUTPUT)
    print("\n[OK] Training complete. Run  python app.py  to start the web server.\n")


if __name__ == "__main__":
    main()
