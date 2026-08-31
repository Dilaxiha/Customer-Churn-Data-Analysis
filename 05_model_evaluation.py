"""
Step 5 - Model Comparison & Evaluation
Loads both models from their subfolders, compares them side-by-side, saves results to a .txt file,
and saves comparison charts inside a dedicated 'model_comparison' subfolder.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_CLEANED, OUTPUT_DIR, RANDOM_STATE, TARGET, TEST_SIZE
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

COMP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "model_comparison")
os.makedirs(COMP_OUTPUT_DIR, exist_ok=True)

# 2. Load data and models
df = pd.read_csv(DATA_CLEANED)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

with open(DT_MODEL_PATH, "rb") as f:
    dt = pickle.load(f)

with open(RF_MODEL_PATH, "rb") as f:
    rf = pickle.load(f)

models = {"Decision Tree": dt, "Random Forest": rf}

# 3. Build comparison table
rows = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv = cross_val_score(model, X, y, cv=5, scoring="f1")
    rows.append(
        {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "CV F1 (mean)": round(cv.mean(), 4),
            "CV F1 (std)": round(cv.std(), 4),
        }
    )

comparison = pd.DataFrame(rows)
comparison_table_str = comparison.to_string(index=False)

# Print comparison table to console
print("\n===== MODEL COMPARISON =====")
print(comparison_table_str)

# 4. Save results to CSV and TXT files
csv_file_path = os.path.join(COMP_OUTPUT_DIR, "model_comparison.csv")
comparison.to_csv(csv_file_path, index=False)

txt_file_path = os.path.join(COMP_OUTPUT_DIR, "model_comparison_results.txt")
save_message = f"All outputs successfully saved to: {COMP_OUTPUT_DIR}"
with open(txt_file_path, "w", encoding="utf-8") as results_file:
    results_file.write("===== MODEL COMPARISON RESULTS =====\n\n")
    results_file.write(comparison_table_str + "\n\n")
    results_file.write(save_message + "\n")

# 5. ROC Curve Comparison
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(COMP_OUTPUT_DIR, "roc_curve_comparison.png"), dpi=150
)
plt.close()

# 6. Confusion Matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Churned", "Churned"]).plot(
        ax=ax
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig(os.path.join(COMP_OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()

print(f"\n{save_message}")