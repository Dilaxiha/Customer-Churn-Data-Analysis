# Key Findings and Conclusion

## Executive Summary

- The churn prediction pipeline used a 3-way split of **70% Train, 15% Validation, and 15% Test** to evaluate model generalization.
- Both models performed strongly, with **Random Forest** showing the best overall accuracy, F1, and ROC-AUC scores.
- There was **no meaningful overfitting** based on the small Train-vs-Validation accuracy gap for either model.
- The strongest churn drivers were **payment delay**, **support calls**, **tenure**, and **usage frequency**.

## Model Performance

| Model | Val Accuracy | Val F1 | Val ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC | CV F1 (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9940 | 0.9937 | 0.9996 | 0.9953 | 0.9951 | 0.9999 | 0.9932 |
| Random Forest | 0.9971 | 0.9969 | 1.0000 | 0.9980 | 0.9979 | 1.0000 | 0.9967 |

- **Decision Tree:** strong and interpretable; validation and test results were both very high.
- **Random Forest:** outperformed the Decision Tree on every key metric and showed the highest cross-validation stability.

## Business Insights

- The most important churn signals were:
  - **Payment Delay**
  - **Support Calls**
  - **Tenure**
  - **Usage Frequency**
- Customers with delayed payments, high support contact, shorter tenure, and lower usage patterns are at the greatest churn risk.

## Conclusion

- **Recommend deploying the Random Forest model** as the primary churn prediction model.
- The Decision Tree remains useful as an interpretable benchmark, but the Random Forest provides the best predictive performance and stability.
- Retention efforts should focus on customers with **late payments, frequent support interactions, lower tenure, and reduced product engagement**.
