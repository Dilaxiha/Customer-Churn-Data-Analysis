# Key Findings and Conclusion

## Executive Summary

- The project objective was to identify customer churn drivers and build a reliable predictive model to support retention and risk management.
- The final modeling pipeline used a 3-way split of 70% Train, 15% Validation, and 15% Test, with stratification to preserve the churn class distribution.
- Both models performed at a very high level, but the Random Forest model showed the strongest overall performance and the highest stability across validation, test, and cross-validation checks.
- The core business takeaway is that churn risk is most strongly associated with payment delays, support contact frequency, tenure, and usage behavior.

## Dataset & Methodology

- Dataset: cleaned customer churn dataset used for supervised classification.
- Target variable: Churn.
- Split strategy:
  - Train: 70%
  - Validation: 15%
  - Test: 15%
- Split method: `train_test_split` with `stratify=y` and `random_state=42`.
- Final split sizes:
  - Train: **45,061 rows**
  - Validation: **9,656 rows**
  - Test: **9,657 rows**
- Evaluation framework:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - Train vs. Validation accuracy gap to assess overfitting
  - 5-fold cross-validation on the training data as a stability check

## Model Performance Breakdown

### Decision Tree

- Overall performance:
  - Train Accuracy: **0.9953**
  - Validation Accuracy: **0.9940**
  - Test Accuracy: **0.9953**
  - Train F1: **0.9951**
  - Validation F1: **0.9937**
  - Test F1: **0.9951**
  - Train ROC-AUC: **0.9999**
  - Validation ROC-AUC: **0.9996**
  - Test ROC-AUC: **0.9999**
- Overfitting assessment:
  - Train vs. Validation accuracy gap: **0.0013**
  - Interpretation: **No significant overfitting detected**.
- Model behavior:
  - The Decision Tree achieved highly competitive performance while remaining interpretable.
  - It correctly classified the majority of churn and non-churn cases across both validation and test data.
- Top features driving churn risk:
  - Payment Delay: **0.4321**
  - Support Calls: **0.1321**
  - Gender: **0.1032**
  - Tenure: **0.0937**
  - Usage Frequency: **0.0911**
- Validation confusion matrix summary:
  - TN: 5,042
  - FP: 40
  - FN: 18
  - TP: 4,556
- Test confusion matrix summary:
  - TN: 5,057
  - FP: 26
  - FN: 19
  - TP: 4,555

### Random Forest

- Overall performance:
  - Train Accuracy: **0.9998**
  - Validation Accuracy: **0.9971**
  - Test Accuracy: **0.9980**
  - Train F1: **0.9998**
  - Validation F1: **0.9969**
  - Test F1: **0.9979**
  - Train ROC-AUC: **1.0000**
  - Validation ROC-AUC: **1.0000**
  - Test ROC-AUC: **1.0000**
- Overfitting assessment:
  - Train vs. Validation accuracy gap: **0.0027**
  - Interpretation: **No significant overfitting detected**.
- Stability checks:
  - 5-fold CV F1 mean: **0.9967**
  - 5-fold CV F1 std: **0.0003**
  - Interpretation: The Random Forest shows exceptional stability and low variance across folds, supporting confidence in generalization.
- Top features driving churn risk:
  - Payment Delay: **0.4539**
  - Support Calls: **0.1676**
  - Tenure: **0.1082**
  - Usage Frequency: **0.0827**
  - Gender: **0.0711**
  - Total Spend: **0.0431**
- Validation confusion matrix summary:
  - TN: 5,077
  - FP: 5
  - FN: 23
  - TP: 4,551
- Test confusion matrix summary:
  - TN: 5,080
  - FP: 3
  - FN: 16
  - TP: 4,558

## Comparative Analysis

| Model | Val Accuracy | Val F1 | Val ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC | CV F1 (mean) | CV F1 (std) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9940 | 0.9937 | 0.9996 | 0.9953 | 0.9951 | 0.9999 | 0.9932 | 0.0005 |
| Random Forest | 0.9971 | 0.9969 | 1.0000 | 0.9980 | 0.9979 | 1.0000 | 0.9967 | 0.0003 |

- The Random Forest model outperformed the Decision Tree on every validation and test metric.
- The validation-test gap was small for both models, indicating robust generalization.
- Random Forest also exhibited the strongest cross-validation stability, with a higher mean F1 and lower standard deviation.
- The Decision Tree remains useful as a transparent benchmark, but it is not the preferred deployment choice when maximum predictive performance is the priority.

## Key Business Insights & Conclusion

- The most consistent churn predictors were:
  - **Payment Delay**
  - **Support Calls**
  - **Tenure**
  - **Usage Frequency**
  - **Gender**
- These variables suggest that customer friction, service dissatisfaction, and low engagement are the most significant churn drivers in the dataset.
- Customers with delayed payments, more frequent support interactions, shorter tenure, and lower or inconsistent usage patterns appear to be at materially higher churn risk.
- The Random Forest model provides the best balance of predictive accuracy, generalization, and cross-validation stability and should be selected for deployment.
- The Decision Tree should be retained as an explanatory model for stakeholder communication and operational review, but not as the primary production classifier when the goal is maximum predictive performance.

### Final Recommendation

- **Deploy the Random Forest model** as the primary churn prediction model for production use.
- Use the Decision Tree model as an auxiliary benchmark for interpretability and business-facing explanation.
- Prioritize retention interventions for customers exhibiting elevated payment delay, frequent support interactions, and shortened tenure or reduced usage patterns.
