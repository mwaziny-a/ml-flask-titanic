import os
import pandas as pd
from sklearn.datasets import fetch_openml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "titanic.csv")

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading Titanic dataset from OpenML...")

# Load Titanic as a pandas DataFrame
titanic = fetch_openml(
    "titanic",
    version=1,
    as_frame=True
)

df = titanic.frame.copy()

print("Downloaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Normalize column names to lowercase for easier handling
df.columns = [col.strip().lower() for col in df.columns]

# Save CSV
df.to_csv(DATA_PATH, index=False, encoding="utf-8")

print(f"Saved to: {DATA_PATH}")
print("First 5 rows:")
print(df.head())