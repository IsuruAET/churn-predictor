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
from openai_service import generate_churn_recommendations, generate_weekly_churn_recommendations

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

# Feature importance mapping
feature_importance_map = {
    "days_since_last_order": "Days Since Last Order",
    "avg_order_count_weekly": "Average Weekly Order Count", 
    "avg_order_count_monthly": "Average Monthly Order Count",
    "avg_order_total_weekly": "Average Weekly Order Total",
    "avg_order_total_monthly": "Average Monthly Order Total"
}

weekly_feature_importance_map = {
    "last12weeks_order_count": "Last 12 Weeks Order Count",
    "last12weeks_order_total": "Last 12 Weeks Order Total",
    "last12weeks_discount_total": "Last 12 Weeks Discount Total",
    "last12weeks_loyalty_earned": "Last 12 Weeks Loyalty Earned"
}

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
        
        # 5. Get feature importance (using permutation importance or feature coefficients)
        if hasattr(model, 'feature_importances_'):
            # For tree-based models (Random Forest, XGBoost, etc.)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models (Logistic Regression, etc.)
            importances = abs(model.coef_[0])
        else:
            # Fallback: use permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, df, [pred], n_repeats=10, random_state=42)
            importances = result.importances_mean
        
        # 6. Get top 2 features
        feature_importance_pairs = list(zip(numeric, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance_pairs[:2]
        
        # 7. Format response
        top_contributors = []
        for feature, importance in top_features:
            top_contributors.append({
                "feature": feature_importance_map.get(feature, feature),
                "importance": float(importance)
            })
        
        # 8. Generate recommendations if churn risk is high
        recommendations = ""
        if pred == 1:
            recommendations = generate_churn_recommendations(top_contributors, data.model_dump())
        
        return {
            "churn_prediction": pred,
            "top_contributing_factors": top_contributors,
            "recommendations": recommendations
        }

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
        
        # 7. Get feature importance for weekly model
        if hasattr(weekly_model, 'feature_importances_'):
            importances = weekly_model.feature_importances_
        elif hasattr(weekly_model, 'coef_'):
            importances = abs(weekly_model.coef_[0])
        else:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(weekly_model, df_scaled, predictions, n_repeats=10, random_state=42)
            importances = result.importances_mean
        
        # 8. Get top 2 features for weekly model
        feature_importance_pairs = list(zip(weekly_numeric, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance_pairs[:2]
        
        # 9. Add top contributing factors to output
        churned_customers['top_factor_1'] = weekly_feature_importance_map.get(top_features[0][0], top_features[0][0])
        churned_customers['top_factor_2'] = weekly_feature_importance_map.get(top_features[1][0], top_features[1][0])
        
        # 10. Generate recommendations for weekly churn
        top_factors_list = [top_features[0][0], top_features[1][0]]
        weekly_recommendations = generate_weekly_churn_recommendations(top_factors_list)
        
        # 11. Select required columns
        output_columns = ['customer_id', 'last12weeks_order_count', 'last12weeks_order_total', 
                         'last12weeks_discount_total', 'last12weeks_loyalty_earned', 
                         'top_factor_1', 'top_factor_2']
        churned_customers = churned_customers[output_columns]
        
        # 11. Convert to CSV
        out = io.StringIO()
        churned_customers.to_csv(out, index=False)
        out.seek(0)
        
        return {
            "csv_data": out.getvalue(),
            "recommendations": weekly_recommendations,
            "churn_count": len(churned_customers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

