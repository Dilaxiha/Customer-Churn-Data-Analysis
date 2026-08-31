import os
import sqlite3
import pandas as pd
from config import DATA_RAW

# 1. Load CSV into SQLite in-memory database
df = pd.read_csv(DATA_RAW)
df.columns = df.columns.str.strip()
conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")

# 2. Define the 15 SQL queries
QUERIES = {
    "1. Total Customers": """
        SELECT COUNT(*) AS total_customers 
        FROM customers;
    """,
    "2. Churn Breakdown": """
        SELECT Churn, COUNT(*) AS count 
        FROM customers 
        GROUP BY Churn;
    """,
    "3. Age Distribution by Churn": """
        SELECT Churn, 
               ROUND(AVG(Age), 2) AS avg_age, 
               MIN(Age) AS min_age, 
               MAX(Age) AS max_age 
        FROM customers 
        GROUP BY Churn;
    """,
    "4. Churn Rate by Subscription Type": """
        SELECT [Subscription Type], 
               COUNT(*) AS total, 
               SUM(Churn) AS churned, 
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct 
        FROM customers 
        GROUP BY [Subscription Type] 
        ORDER BY churn_rate_pct DESC;
    """,
    "5. Churn Rate by Contract Length": """
        SELECT [Contract Length], 
               COUNT(*) AS total, 
               SUM(Churn) AS churned, 
               ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct 
        FROM customers 
        GROUP BY [Contract Length] 
        ORDER BY churn_rate_pct DESC;
    """,
    "6. Spend by Churn Status": """
        SELECT Churn, 
               ROUND(AVG([Total Spend]), 2) AS avg_spend, 
               ROUND(SUM([Total Spend]), 2) AS total_spend 
        FROM customers 
        GROUP BY Churn;
    """,
    "7. Top 10 Spenders": """
        SELECT CustomerID, Age, Gender, [Total Spend], Churn 
        FROM customers 
        ORDER BY [Total Spend] DESC 
        LIMIT 10;
    """,
    "8. Average Support Calls by Churn": """
        SELECT Churn, 
               ROUND(AVG([Support Calls]), 2) AS avg_support_calls 
        FROM customers 
        GROUP BY Churn;
    """,
    "9. Churn by Support Volume": """
        SELECT CASE 
                   WHEN [Support Calls] > 5 THEN 'High' 
                   ELSE 'Low' 
               END AS support_level, 
               COUNT(*) AS total, 
               SUM(Churn) AS churned, 
               ROUND(AVG(Churn)*100, 2) AS churn_rate_pct 
        FROM customers 
        GROUP BY support_level;
    """,
    "10. Churn by Usage Frequency": """
        SELECT CASE 
                   WHEN [Usage Frequency] <= 10 THEN '0-10' 
                   WHEN [Usage Frequency] <= 20 THEN '11-20' 
                   ELSE '21+' 
               END AS usage_bucket, 
               COUNT(*) AS total, 
               ROUND(AVG(Churn)*100, 2) AS churn_rate_pct 
        FROM customers 
        GROUP BY usage_bucket 
        ORDER BY usage_bucket;
    """,
    "11. Average Tenure by Churn": """
        SELECT Churn, 
               ROUND(AVG(Tenure), 2) AS avg_tenure 
        FROM customers 
        GROUP BY Churn;
    """,
    "12. Churn Rate by Tenure Group": """
        SELECT CASE 
                   WHEN Tenure < 12 THEN 'Short (<12m)' 
                   ELSE 'Long (12m+)' 
               END AS tenure_group, 
               COUNT(*) AS total, 
               SUM(Churn) AS churned, 
               ROUND(AVG(Churn)*100, 2) AS churn_rate_pct 
        FROM customers 
        GROUP BY tenure_group;
    """,
    "13. Churn Rate by Gender": """
        SELECT Gender, 
               COUNT(*) AS total, 
               SUM(Churn) AS churned, 
               ROUND(AVG(Churn)*100, 2) AS churn_rate_pct 
        FROM customers 
        GROUP BY Gender;
    """,
    "14. Payment Delay by Churn": """
        SELECT Churn, 
               ROUND(AVG([Payment Delay]), 2) AS avg_payment_delay, 
               MAX([Payment Delay]) AS max_delay 
        FROM customers 
        GROUP BY Churn;
    """,
    "15. High-Risk Customer Count": """
        SELECT COUNT(*) AS high_risk_count, 
               SUM(Churn) AS churned, 
               ROUND(AVG(Churn)*100, 2) AS churn_rate_pct 
        FROM customers 
        WHERE [Support Calls] >= 7 
          AND Tenure < 15 
          AND [Payment Delay] >= 20;
    """,
}

# 3. Create 'output' folder if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, "sql_query_results.txt")

# 4. Run queries, print to console, and write results to the file
with open(output_file_path, "w") as f:
    for title, sql in QUERIES.items():
        header = f"{'=' * 65}\n{title}\n{'-' * 65}\n"
        print(header, end="")
        f.write(header)
        
        result = pd.read_sql_query(sql, conn)
        result_str = result.to_string(index=False) + "\n\n"
        
        print(result_str)
        f.write(result_str)

conn.close()
print(f"All 15 SQL queries executed successfully.")
print(f"Results saved to file: {output_file_path}")