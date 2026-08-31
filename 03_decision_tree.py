"""
Step 3 - Decision Tree Classifier
Trains a Decision Tree on the cleaned data, saves the model.
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
from sklearn.tree import DecisionTreeClassifier, plot_tree


DECISION_TREE_DIR = os.path.join(OUTPUT_DIR, "decision_tree")
os.makedirs(DECISION_TREE_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(DECISION_TREE_DIR, "decision_tree_results.txt")
FEATURE_IMPORTANCE_PATH = os.path.join(
    DECISION_TREE_DIR, "dt_feature_importance.png"
)
TREE_PLOT_PATH = os.path.join(DECISION_TREE_DIR, "dt_tree_plot.png")
MODEL_PATH = os.path.join(DECISION_TREE_DIR, "decision_tree_model.pkl")

output_buffer = io.StringIO()
with redirect_stdout(output_buffer):
    # Load cleaned data
    df = pd.read_csv(DATA_CLEANED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create and train the Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=8, min_samples_split=20, random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)

    # Make predictions on train and test data to assess potential overfitting
    y_train_pred = dt.predict(X_train)
    y_train_prob = dt.predict_proba(X_train)[:, 1]
    y_pred = dt.predict(X_test)
    y_prob = dt.predict_proba(X_test)[:, 1]

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

    # Print evaluation metrics
    print("\n===== DECISION TREE RESULTS =====")
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

results = output_buffer.getvalue()
print(results, end="")
with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
    results_file.write(results)

# Feature importance chart
feat_imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(
    ascending=True
)
plt.figure(figsize=(10, 6))
feat_imp.plot.barh()
plt.title("Decision Tree - Feature Importance")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
plt.close()

# Visualize the tree (top 4 levels)
plt.figure(figsize=(24, 10))
plot_tree(
    dt,
    max_depth=4,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree (top 4 levels)")
plt.tight_layout()
plt.savefig(TREE_PLOT_PATH, dpi=150)
plt.close()

# Save model to disk
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(dt, model_file)