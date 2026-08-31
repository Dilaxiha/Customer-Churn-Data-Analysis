"""
Step 2 - Data Cleaning
Handles missing values, duplicates, outliers, type fixes.
Saves cleaned CSV to data_cleaned.csv.
"""

import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_CLEANED, TARGET


# Load raw data
df = pd.read_csv(DATA_RAW)
df.columns = df.columns.str.strip()

print("Raw shape:", df.shape)


# 1. Drop duplicates
before = len(df)

df.drop_duplicates(inplace=True)

print(f"Duplicates removed: {before - len(df)}")


# 2. Handle missing values
missing = df.isnull().sum()

print("\nMissing values per column:")

if missing.any():
    print(missing[missing > 0])
else:
    print("None")


# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()


# Fill missing numeric values with median
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())


# Fill missing categorical values with mode
for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])


# 3. Outlier analysis using the 1st and 99th percentiles
# Extreme values are investigated rather than automatically removed.
# Total Spend is retained because all values fall within the valid
# dataset range of $100-$1,000.

outlier_cols = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

print("\nOutlier analysis (1st-99th percentile):")

for col in outlier_cols:
    lower_percentile = df[col].quantile(0.01)
    upper_percentile = df[col].quantile(0.99)

    outliers = (
        (df[col] < lower_percentile) |
        (df[col] > upper_percentile)
    ).sum()

    print(
        f"{col}: "
        f"Lower={lower_percentile:.2f}, "
        f"Upper={upper_percentile:.2f}, "
        f"Count={outliers}"
    )


# 4. Encode categorical features

# Encode Gender
df["Gender"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})


# One-hot encode Subscription Type and Contract Length
df = pd.get_dummies(
    df,
    columns=["Subscription Type", "Contract Length"],
    drop_first=True,
    dtype=int
)


# 5. Drop CustomerID because it is not a useful predictive feature
df.drop(columns=["CustomerID"], inplace=True)


# 6. Save cleaned dataset
df.to_csv(DATA_CLEANED, index=False)

print(f"\nCleaned shape: {df.shape}")
print(f"Saved to: {DATA_CLEANED}")