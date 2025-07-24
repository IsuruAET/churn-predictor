# main.py
"""
FastAPI endpoint for customer-churn prediction.
Expects 5 numeric features.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib, json, pandas as pd

# --------------------------------------------------------------
# Load artefacts
# --------------------------------------------------------------
model   = joblib.load("models/churn_model.pkl")
scaler  = joblib.load("models/scaler.pkl")
numeric = json.load(open("models/numeric_columns.json"))  # 5 numeric columns

# --------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn for mosaic customers using 5 behavioral features",
    version="2.0"
)

# --------------------------------------------------------------
# Input schema
# --------------------------------------------------------------
class CustomerData(BaseModel):
    days_since_last_order:     float
    avg_order_count_weekly:    float
    avg_order_count_monthly:   float
    avg_order_total_weekly:    float
    avg_order_total_monthly:   float

# --------------------------------------------------------------
# Prediction endpoint
# --------------------------------------------------------------
@app.post("/predict/")
def predict_churn(data: CustomerData):
    try:
        # 1. to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 2. Ensure correct column order
        df = df[numeric]

        # 3. Scale numeric features
        df[numeric] = scaler.transform(df[numeric])

        # 4. Predict
        pred = int(model.predict(df)[0])
        return {"churn_prediction": pred}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict/")
def batch_predict_churn(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        # Read CSV into DataFrame
        df = pd.read_csv(file.file)
        # Ensure correct columns
        if set(numeric) - set(df.columns):
            raise HTTPException(status_code=400, detail=f"Missing columns: {set(numeric) - set(df.columns)}")
        df = df[numeric]
        # Scale
        df[numeric] = scaler.transform(df[numeric])
        # Predict
        preds = model.predict(df)
        return {"churn_predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
