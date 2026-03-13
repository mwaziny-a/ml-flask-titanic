import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ========= 1. Paths =========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "titanic.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

print("BASE_DIR:", BASE_DIR)
print("DATA_PATH:", DATA_PATH)
print("DATA exists:", os.path.exists(DATA_PATH))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

if os.path.getsize(DATA_PATH) == 0:
    raise ValueError(f"Dataset file is empty: {DATA_PATH}")


# ========= 2. Load data =========
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print("Error while reading CSV:", str(e))
    raise

print("\nDataset loaded successfully")
print("Dataset shape:", df.shape)
print("Columns found:", df.columns.tolist())


# ========= 3. Normalize column names =========
df.columns = [col.strip().lower() for col in df.columns]

print("Normalized columns:", df.columns.tolist())

required_map = {
    "survived": ["survived"],
    "pclass": ["pclass"],
    "sex": ["sex"],
    "age": ["age"],
    "sibsp": ["sibsp"],
    "parch": ["parch"],
    "fare": ["fare"],
    "embarked": ["embarked"]
}

missing_needed = [key for key, aliases in required_map.items() if not any(a in df.columns for a in aliases)]
if missing_needed:
    raise ValueError(f"Missing required Titanic columns: {missing_needed}")


# ========= 4. Keep needed columns =========
selected_columns = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = df[selected_columns].copy()

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Target cleanup
df = df.dropna(subset=["survived"])
df["survived"] = df["survived"].astype(int)

target_column = "survived"
feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

X = df[feature_columns]
y = df[target_column]


# ========= 5. Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ========= 6. Feature groups =========
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]


# ========= 7. Preprocessing =========
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ========= 8. Full pipeline =========
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ========= 9. Train =========
pipeline.fit(X_train, y_train)


# ========= 10. Evaluate =========
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))


# ========= 11. Save model =========
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)

metadata = {
    "project_name": "Titanic Survival Prediction",
    "task_type": "classification",
    "dataset_file": "data/titanic.csv",
    "target": target_column,
    "features": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_type": "LogisticRegression",
    "metrics": {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4)
    }
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved to:", MODEL_PATH)
print("Metadata saved to:", METADATA_PATH)