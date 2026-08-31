"""Central configuration for paths and constants."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the raw CSV file
DATA_RAW = os.path.join(BASE_DIR, "customer_churn_dataset-testing-master.csv")

# Path to output directory and clean data
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_CLEANED = os.path.join(OUTPUT_DIR, "data_cleaned.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Project constants
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2