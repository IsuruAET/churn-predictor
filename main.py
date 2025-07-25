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
from db_config import get_db_engine

# --------------------------------------------------------------
# Load artefacts
# --------------------------------------------------------------
model   = joblib.load("models/churn_model.pkl")
scaler  = joblib.load("models/scaler.pkl")
numeric = json.load(open("models/numeric_columns.json"))  # 5 numeric columns

# Weekly model artefacts
weekly_model   = joblib.load("models/churn_model_weekly.pkl")
weekly_scaler  = joblib.load("models/scaler_weekly.pkl")
weekly_numeric = json.load(open("models/numeric_columns_weekly.json"))  # 4 numeric columns

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

# --------------------------------------------------------------
# Next week churn prediction endpoint
# --------------------------------------------------------------
def fetch_customer_weekly_data(engine):
    query = """
    SELECT
        company_id,
        customer_id,
        SUM(order_count) AS last12weeks_order_count,
        SUM(order_total) AS last12weeks_order_total,
        SUM(discount_total) AS last12weeks_discount_total,
        SUM(loyalty_earned) AS last12weeks_loyalty_earned
    FROM customer_tx_weekly
    WHERE week_end_date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
    GROUP BY company_id, customer_id
    ORDER BY company_id, customer_id;
    """
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

@app.post("/next-week-predict/")
def next_week_churn_predict():
    try:
        # 1. Fetch customer weekly data
        engine = get_db_engine()
        df = fetch_customer_weekly_data(engine)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No customer data found")
        
        # 2. Keep original data for output
        original = df.copy()
        
        # 3. Prepare features for prediction
        df_features = df[weekly_numeric]
        
        # 4. Scale features
        df_scaled = weekly_scaler.transform(df_features)
        
        # 5. Predict churn
        predictions = weekly_model.predict(df_scaled)
        
        # 6. Filter only churned customers (prediction = 1)
        churned_customers = original[predictions == 1].copy()
        churned_customers['churn_prediction'] = predictions[predictions == 1]
        
        # 7. Select required columns
        output_columns = ['customer_id', 'last12weeks_order_count', 'last12weeks_order_total', 
                         'last12weeks_discount_total', 'last12weeks_loyalty_earned']
        churned_customers = churned_customers[output_columns]
        
        # 8. Convert to CSV
        out = io.StringIO()
        churned_customers.to_csv(out, index=False)
        out.seek(0)
        
        return StreamingResponse(
            out, 
            media_type="text/csv", 
            headers={"Content-Disposition": "attachment; filename=next_week_churn_customers.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

