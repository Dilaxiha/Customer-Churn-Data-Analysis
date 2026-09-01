# Key Findings and Conclusion

## Executive Summary

- The churn prediction pipeline used a 3-way split of **70% Train, 15% Validation, and 15% Test** to evaluate model generalization.
- Across four models, **XGBoost** produced the strongest performance, followed closely by **Random Forest**.
- **Logistic Regression** underperformed substantially relative to the tree-based models on this dataset.
- The most important churn drivers were **payment delay**, **support calls**, **tenure**, and **usage frequency**.

## Model Performance

| Model | Val Accuracy | Val F1 | Val ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC | CV F1 (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9995 | 0.9995 | 1.0000 | 0.9996 | 0.9996 | 1.0000 | 0.9995 |
| Random Forest | 0.9971 | 0.9969 | 1.0000 | 0.9980 | 0.9979 | 1.0000 | 0.9967 |
| Decision Tree | 0.9940 | 0.9937 | 0.9996 | 0.9953 | 0.9951 | 0.9999 | 0.9932 |
| Logistic Regression | 0.5947 | 0.3648 | 0.6539 | 0.5930 | 0.3566 | 0.6430 | 0.8180 |

- **XGBoost:** best overall predictive performance and highest stability across validation, test, and cross-validation.
- **Random Forest:** very strong second choice with excellent generalization and interpretability through feature importance.
- **Decision Tree:** competitive and interpretable, but weaker than the top two models.
- **Logistic Regression:** materially weaker performance, indicating nonlinear relationships are important for churn prediction.

## Business Insights

- The most important churn signals were:
  - **Payment Delay**
  - **Support Calls**
  - **Tenure**
  - **Usage Frequency**
- Customers with late payments, high support contact, shorter tenure, and lower usage activity are at the greatest risk of churn.

## Conclusion

- **Recommend deploying the XGBoost model** as the primary churn prediction model.
- **Random Forest** is the best alternative if model interpretability and feature importance are important.
- Retention strategies should prioritize customers with **late payments, frequent service issues, shorter tenure, and reduced engagement**.
