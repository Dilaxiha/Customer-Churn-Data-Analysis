"""
Step 1 - Exploratory Data Analysis (EDA)
Imports configuration settings from config.py, inspects dataset health, 
performs outlier analysis, calculates risk metrics, and saves charts to outputs/.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Import centralized parameters from config.py
from config import DATA_RAW, OUTPUT_DIR, TARGET

# 1. Load Data
df = pd.read_csv(DATA_RAW)
df.columns = df.columns.str.strip()

# Keep integrity dimensions for the source dataset before removing its identifier.
source_shape = df.shape
null_count = int(df.isnull().sum().sum())
duplicate_count = int(df.duplicated().sum())

# Drop CustomerID if it exists so it doesn't affect any statistics or plots
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# 2. Text Summary Calculations
shape_info = f"Dataset shape: {source_shape[0]:,} rows x {source_shape[1]} columns"

# 3. Numeric Columns Selection
num_cols = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

# 4. Outlier Scan (1st - 99th Percentile Method)
outlier_rows = []
for col in num_cols:
    p1 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    outliers = df[(df[col] < p1) | (df[col] > p99)]
    outlier_rows.append({
        'Variable': col,
        '1st %ile': p1,
        '99th %ile': p99,
        'Outlier Count': len(outliers),
        'Outlier Pct (%)': round((len(outliers) / len(df)) * 100, 2)
    })

outlier_summary_df = pd.DataFrame(outlier_rows)

# 5. Target Column Balance
churn_counts = df[TARGET].value_counts()
churn_percentages = df[TARGET].value_counts(normalize=True) * 100
churn_dist_df = pd.DataFrame({'Count': churn_counts, 'Percentage': churn_percentages})

# 6. Descriptive Statistics
num_stats = df[num_cols].agg(['mean', 'median', 'min', 'max', 'std']).T

# 7. Categorical Breakdowns (Normalized Share Summing to 100%)
total_churned = df[df[TARGET] == 1].shape[0]

def build_cat_table(data, col):
    counts = data[col].value_counts()
    churn_rates = data.groupby(col)[TARGET].mean() * 100
    churn_shares = (data[data[TARGET] == 1][col].value_counts() / total_churned) * 100
    
    return pd.DataFrame({
        'Count': counts,
        'Churn Rate (%)': churn_rates,
        'Share of Total Churn (%)': churn_shares
    })

cat_cols = ["Gender", "Subscription Type", "Contract Length"]
cat_tables = {col: build_cat_table(df, col) for col in cat_cols}

# 8. Risk Drivers
risk_features = ['Support Calls', 'Payment Delay', 'Total Spend']
risk_0 = df[df[TARGET] == 0][risk_features].mean()
risk_1 = df[df[TARGET] == 1][risk_features].mean()
risk_df = pd.DataFrame({
    'Churn = 0': risk_0,
    'Churn = 1': risk_1,
    'Mean Difference': risk_1 - risk_0
})

# 9. Export Text Summary Report
output_text_file = os.path.join(OUTPUT_DIR, 'exploration_summary.txt')
with open(output_text_file, 'w') as f:
    f.write("CUSTOMER CHURN DATA EXPLORATION\n")
    f.write("================================\n\n")
    f.write("1. DATA INTEGRITY\n")
    f.write(f"{shape_info}\n")
    f.write(f"Null values: {null_count}\n")
    f.write(f"Duplicate rows: {duplicate_count}\n\n")
    f.write("2. OUTLIER ANALYSIS ACROSS ALL VARIABLES (1st-99th Percentile)\n")
    f.write(f"{outlier_summary_df.to_string(index=False)}\n\n")
    f.write("3. CHURN DISTRIBUTION\n")
    f.write(f"{churn_dist_df.to_string()}\n\n")
    f.write("4. SUMMARY STATISTICS\n")
    f.write(f"{num_stats.to_string()}\n\n")
    for col, table in cat_tables.items():
        f.write(f"5. {col.upper()} BREAKDOWN\n")
        f.write(f"{table.to_string()}\n\n")
    f.write("6. RISK METRICS AND SEGMENTATION\n")
    f.write(f"{risk_df.to_string()}\n")

# 10. Plotting & Exporting Visualizations (Enhanced Aesthetics)
# Set global aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})

# --- A. Numeric Histograms ---
n_num = len(num_cols)
n_cols = 3
n_rows = (n_num + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.8 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.histplot(
        df[col], 
        ax=ax, 
        color="#2b5c8f", 
        edgecolor="white", 
        linewidth=0.5
    )
    sns.kdeplot(df[col], ax=ax, linewidth=2, color="#d95f02")
    ax.set_title(f"Distribution of {col}", fontweight="bold", pad=10)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    sns.despine(ax=ax)

# Remove extra empty subplots
for i in range(n_num, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "numeric_distributions.png"), dpi=300)
plt.close()

# --- B. Categorical Bar Charts ---
fig, axes = plt.subplots(1, len(cat_cols), figsize=(5.5 * len(cat_cols), 4.5))
if len(cat_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, cat_cols):
    counts = df[col].value_counts()
    total = len(df)
    palette = sns.color_palette("Blues_r", n_colors=len(counts))
    
    bars = sns.barplot(
        x=counts.index,
        y=counts.values,
        ax=ax,
        palette=palette,
        hue=counts.index,
        legend=False,
        edgecolor="black",
        linewidth=0.5,
    )
    
    # Add count and percentage labels above bars
    for bar, count in zip(bars.patches, counts.values):
        count = int(count)
        pct = (count / total) * 100
        ax.annotate(
            f"{count:,}\n({pct:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

    ax.set_title(f"{col} Counts", fontweight="bold", pad=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Customer Count")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, counts.max() * 1.2)  # Make room for bar labels
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "categorical_distributions.png"), dpi=300)
plt.close()

# --- C. Correlation Heatmap ---
plt.figure(figsize=(10, 8))
mask = None  # Set to np.triu(np.ones_like(df[num_cols].corr(), dtype=bool)) if you prefer half-matrix

sns.heatmap(
    df[num_cols].corr(), 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    center=0,
    square=True, 
    linewidths=0.8, 
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    annot_kws={"size": 9}
)

plt.title("Feature Correlation Matrix", fontweight="bold", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=300)
plt.close()

print(f"EDA complete. Enhanced visualization charts saved to '{OUTPUT_DIR}/'.")