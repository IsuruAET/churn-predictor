# main.py
"""
FastAPI endpoint for customer-churn prediction.
Expects 5 numeric features.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib, json, pandas as pd
from fastapi.responses import StreamingResponse
import io

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

# --------------------------------------------------------------
# Batch prediction endpoint
# --------------------------------------------------------------
@app.post("/batch-predict/")
def batch_predict_churn(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV into DataFrame
        df = pd.read_csv(file.file)
        original = df.copy()  # keep all columns
        # Ensure correct columns for prediction
        df_numeric = df[numeric]
        # Scale
        df_numeric = scaler.transform(df_numeric)
        # Predict
        preds = model.predict(df_numeric)
        original["churn_prediction"] = preds
        # Convert to CSV in-memory
        out = io.StringIO()
        original.to_csv(out, index=False)
        out.seek(0)
        return StreamingResponse(out, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

