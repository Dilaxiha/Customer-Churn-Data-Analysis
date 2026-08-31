"""
Step 4 - Random Forest Classifier
Trains a Random Forest on the cleaned data, saves evaluation metrics to a .txt file,
and saves plots/models inside a dedicated 'random_forest' output subfolder.
"""

import io
import os
import pickle
from contextlib import redirect_stdout

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_CLEANED, OUTPUT_DIR, RANDOM_STATE, TARGET, TEST_SIZE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# 1. Setup dedicated subfolder under OUTPUT_DIR
RF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "random_forest")
os.makedirs(RF_OUTPUT_DIR, exist_ok=True)

results_buffer = io.StringIO()
with redirect_stdout(results_buffer):
    # Load cleaned data
    df = pd.read_csv(DATA_CLEANED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create and train the Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Make predictions on train and test data to assess potential overfitting
    y_train_pred = rf.predict(X_train)
    y_train_prob = rf.predict_proba(X_train)[:, 1]
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_pred)
    accuracy_gap = train_accuracy - test_accuracy
    overfitting_status = (
        "Likely overfit"
        if accuracy_gap > 0.05
        else "No significant overfitting detected"
    )

    print("\n===== RANDOM FOREST RESULTS =====")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Accuracy Gap   : {accuracy_gap:.4f}")
    print(f"Train F1       : {train_f1:.4f}")
    print(f"Test F1        : {test_f1:.4f}")
    print(f"Train ROC-AUC  : {roc_auc_score(y_train, y_train_prob):.4f}")
    print(f"Test ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Overfitting    : {overfitting_status}")

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report (Test Set):")
    print(
        classification_report(
            y_test, y_pred, target_names=["Not Churned", "Churned"]
        )
    )

    # Save feature importance plot
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=True
    )
    plt.figure(figsize=(10, 6))
    feat_imp.plot.barh()
    plt.title("Random Forest - Feature Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RF_OUTPUT_DIR, "rf_feature_importance.png"), dpi=150
    )
    plt.close()

    # Save model to disk
    model_file_path = os.path.join(RF_OUTPUT_DIR, "random_forest_model.pkl")
    with open(model_file_path, "wb") as model_file:
        pickle.dump(rf, model_file)

    print(f"All outputs successfully saved to: {RF_OUTPUT_DIR}")

results = results_buffer.getvalue()
with open(
    os.path.join(RF_OUTPUT_DIR, "random_forest_results.txt"),
    "w",
    encoding="utf-8",
) as results_file:
    results_file.write(results)

print(results, end="")