# Titanic Survival Prediction Web App

This project is an end-to-end machine learning classification web app built with Flask and scikit-learn.
It predicts whether a passenger would likely survive the Titanic disaster based on passenger information.

## Dataset
- Dataset: Titanic dataset
- Task Type: Classification
- Target Column: survived

## Features Used
- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

## Model
- Logistic Regression
- Built using a full scikit-learn Pipeline:
  - Missing value imputation
  - Numeric scaling
  - One-hot encoding
  - Classification model

## Evaluation Metrics
After training, the script prints:
- Accuracy
- Precision
- Recall
- F1-score

## Project Structure
ml-flask-titanic/
│
├── src/
│   ├── train.py
│   └── app_flask.py
├── model/
│   ├── pipeline.joblib
│   └── metadata.json
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── data/
│   └── titanic.csv
├── requirements.txt
├── .gitignore
└── README.md

## Setup Instructions

### 1. Clone the repository
git clone <YOUR_REPO_URL>
cd ml-flask-titanic

### 2. Create a virtual environment
python -m venv .venv

### 3. Activate it
Windows:
.venv\Scripts\activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Train the model
python src/train.py

### 6. Run the Flask app
python src/app_flask.py

## Deployment
Recommended deployment platform:
- Render

Start command:
gunicorn src.app_flask:app

## Screenshots
Add 2–3 screenshots here:
- Input form
- Prediction result