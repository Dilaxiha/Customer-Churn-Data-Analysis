"""
Step 5 - Model Comparison & Evaluation
Loads Decision Tree and Random Forest models, computes evaluation metrics on 
Validation and Test sets, saves comparison artifacts, and exports ROC/Confusion Matrix plots.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_CLEANED, OUTPUT_DIR, RANDOM_STATE, TARGET
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split

# 1. Setup output paths
DT_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "decision_tree", "decision_tree_model.pkl"
)
RF_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "random_forest", "random_forest_model.pkl"
)
LR_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "logistic_regression", "logistic_regression_model.pkl"
)
XGB_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "xgboost", "xgboost_model.pkl"
)

COMP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "model_comparison")
os.makedirs(COMP_OUTPUT_DIR, exist_ok=True)

# 2. Load data and models
df = pd.read_csv(DATA_CLEANED)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 3-way split: 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

with open(DT_MODEL_PATH, "rb") as f:
    dt = pickle.load(f)

with open(RF_MODEL_PATH, "rb") as f:
    rf = pickle.load(f)

with open(LR_MODEL_PATH, "rb") as f:
    lr = pickle.load(f)

with open(XGB_MODEL_PATH, "rb") as f:
    xgb = pickle.load(f)

models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "Logistic Regression": lr,
    "XGBoost": xgb,
}

with open(os.path.join(OUTPUT_DIR, "logistic_regression", "scaler.pkl"), "rb") as scaler_file:
    lr_scaler = pickle.load(scaler_file)

X_train_scaled = lr_scaler.transform(X_train)
X_val_scaled = lr_scaler.transform(X_val)
X_test_scaled = lr_scaler.transform(X_test)

# 3. Build side-by-side metric comparison table
rows = []
for name, model in models.items():
    if name == "Logistic Regression":
        X_train_model = X_train_scaled
        X_val_model = X_val_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_val_model = X_val
        X_test_model = X_test

    val_pred = model.predict(X_val_model)
    val_prob = model.predict_proba(X_val_model)[:, 1]
    test_pred = model.predict(X_test_model)
    test_prob = model.predict_proba(X_test_model)[:, 1]
    cv = cross_val_score(model, X_train_model, y_train, cv=5, scoring="f1")

    rows.append(
        {
            "Model": name,
            "Val Acc": round(accuracy_score(y_val, val_pred), 4),
            "Test Acc": round(accuracy_score(y_test, test_pred), 4),
            "Val Prec": round(precision_score(y_val, val_pred), 4),
            "Test Prec": round(precision_score(y_test, test_pred), 4),
            "Val Rec": round(recall_score(y_val, val_pred), 4),
            "Test Rec": round(recall_score(y_test, test_pred), 4),
            "Val F1": round(f1_score(y_val, val_pred), 4),
            "Test F1": round(f1_score(y_test, test_pred), 4),
            "Val AUC": round(roc_auc_score(y_val, val_prob), 4),
            "Test AUC": round(roc_auc_score(y_test, test_prob), 4),
            "CV F1 (Mean)": round(cv.mean(), 4),
            "CV F1 (Std)": round(cv.std(), 4),
        }
    )

comparison = pd.DataFrame(rows)
comparison_table_str = comparison.to_string(index=False)

print("\n===== MODEL COMPARISON =====")
print(comparison_table_str)

# 4. Save metrics to disk
csv_file_path = os.path.join(COMP_OUTPUT_DIR, "model_comparison.csv")
comparison.to_csv(csv_file_path, index=False)

txt_file_path = os.path.join(COMP_OUTPUT_DIR, "model_comparison_results.txt")
save_message = f"All outputs successfully saved to: {COMP_OUTPUT_DIR}"
with open(txt_file_path, "w", encoding="utf-8") as results_file:
    results_file.write("===== MODEL COMPARISON RESULTS =====\n\n")
    results_file.write(comparison_table_str + "\n\n")
    results_file.write(save_message + "\n")

# 5. Plot ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (label, X_eval, y_eval) in zip(
    axes, [("Validation Set", X_val, y_val), ("Test Set", X_test, y_test)]
):
    for name, model in models.items():
        if name == "Logistic Regression":
            X_eval_model = lr_scaler.transform(X_eval)
        else:
            X_eval_model = X_eval

        y_prob = model.predict_proba(X_eval_model)[:, 1]
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
        auc = roc_auc_score(y_eval, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {label}")
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(COMP_OUTPUT_DIR, "roc_curve_comparison.png"), dpi=150)
plt.close()

# 6. Plot Confusion Matrices
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
for row, (label, X_eval, y_eval) in enumerate(
    [("Validation", X_val, y_val), ("Test", X_test, y_test)]
):
    for col, (name, model) in enumerate(models.items()):
        if name == "Logistic Regression":
            X_eval_model = lr_scaler.transform(X_eval)
        else:
            X_eval_model = X_eval

        y_pred = model.predict(X_eval_model)
        cm = confusion_matrix(y_eval, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Not Churned", "Churned"]).plot(
            ax=axes[row, col], cmap="Blues", colorbar=False
        )
        axes[row, col].set_title(f"{label} - {name}")

plt.tight_layout()
plt.savefig(os.path.join(COMP_OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()

print(f"\n{save_message}")