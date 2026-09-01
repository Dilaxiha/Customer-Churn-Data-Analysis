"""
Step 6 - Logistic Regression Classifier
Trains a Logistic Regression model on the cleaned data with a Train / Validation / Test split,
evaluates performance across all three sets, and saves the model artifacts.
"""

import io
import os
import pickle
from contextlib import redirect_stdout

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_CLEANED, OUTPUT_DIR, RANDOM_STATE, TARGET
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "logistic_regression")
os.makedirs(LR_OUTPUT_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(LR_OUTPUT_DIR, "logistic_regression_results.txt")
COEFFICIENT_PATH = os.path.join(LR_OUTPUT_DIR, "lr_feature_coefficients.png")
MODEL_PATH = os.path.join(LR_OUTPUT_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(LR_OUTPUT_DIR, "scaler.pkl")

output_buffer = io.StringIO()
with redirect_stdout(output_buffer):
    df = pd.read_csv(DATA_CLEANED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 1. Split into Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Split Temp into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\n=== DATA SPLIT SUMMARY ===")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]

    y_val_pred = model.predict(X_val_scaled)
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_prob)
    val_auc = roc_auc_score(y_val, y_val_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)

    train_val_gap = train_acc - val_acc
    overfitting_status = (
        "Likely overfit (Train vs Validation gap > 5%)"
        if train_val_gap > 0.05
        else "No significant overfitting detected"
    )

    print("\n===== LOGISTIC REGRESSION RESULTS =====")
    print(f"Train Accuracy      : {train_acc:.4f}")
    print(f"Validation Accuracy : {val_acc:.4f}")
    print(f"Test Accuracy       : {test_acc:.4f}")
    print(f"Train vs Val Gap    : {train_val_gap:.4f}")
    print(f"Train F1            : {train_f1:.4f}")
    print(f"Validation F1       : {val_f1:.4f}")
    print(f"Test F1             : {test_f1:.4f}")
    print(f"Train ROC-AUC       : {train_auc:.4f}")
    print(f"Validation ROC-AUC  : {val_auc:.4f}")
    print(f"Test ROC-AUC        : {test_auc:.4f}")
    print(f"Overfitting Status  : {overfitting_status}")

    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    print("\nClassification Report (Validation Set):")
    print(
        classification_report(
            y_val, y_val_pred, target_names=["Not Churned", "Churned"]
        )
    )

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (Test Set):")
    print(
        classification_report(
            y_test, y_test_pred, target_names=["Not Churned", "Churned"]
        )
    )

    coef = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    coef.plot.barh()
    plt.title("Logistic Regression - Feature Coefficients")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(COEFFICIENT_PATH, dpi=150)
    plt.close()

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(SCALER_PATH, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"\nAll outputs successfully saved to: {LR_OUTPUT_DIR}")

results = output_buffer.getvalue()
print(results, end="")
with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
    results_file.write(results)
