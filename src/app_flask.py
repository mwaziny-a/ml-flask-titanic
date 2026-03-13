import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pipeline.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model", "metadata.json")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

pipeline = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text=None,
        probability_text=None,
        error_text=None,
        metadata=metadata
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_df = pd.DataFrame([{
            "pclass": int(request.form["pclass"]),
            "sex": request.form["sex"],
            "age": float(request.form["age"]),
            "sibsp": int(request.form["sibsp"]),
            "parch": int(request.form["parch"]),
            "fare": float(request.form["fare"]),
            "embarked": request.form["embarked"]
        }])

        prediction = pipeline.predict(input_df)[0]

        probability_text = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0][1]
            probability_text = f"{proba * 100:.2f}%"

        result = "Passenger likely survived" if prediction == 1 else "Passenger likely did NOT survive"

        return render_template(
            "index.html",
            prediction_text=result,
            probability_text=probability_text,
            error_text=None,
            metadata=metadata
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=None,
            probability_text=None,
            error_text=str(e),
            metadata=metadata
        )