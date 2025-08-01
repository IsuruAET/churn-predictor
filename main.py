# main.py
"""
FastAPI endpoint for customer-churn prediction.
Expects 12 numeric features.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib, json, pandas as pd
from fastapi.responses import StreamingResponse
import io
from db_config import get_db_engine
from openai_service import generate_churn_recommendations, generate_weekly_churn_recommendations
import numpy as np
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(verbose=True)

# --------------------------------------------------------------
# Load artefacts
# --------------------------------------------------------------
# Time series model artefacts
time_series_model = joblib.load("models/churn_model_time_series.pkl")
time_series_scaler = joblib.load("models/scaler_time_series.pkl")
time_series_feature_names = joblib.load("models/feature_names_time_series.pkl")

# Time series feature importance mapping
time_series_feature_importance_map = {
    "week_0": "Most Recent Week Order Count",
    "week_1": "Week -1 Order Count",
    "week_2": "Week -2 Order Count",
    "week_3": "Week -3 Order Count",
    "week_4": "Week -4 Order Count",
    "week_5": "Week -5 Order Count",
    "week_6": "Week -6 Order Count",
    "week_7": "Week -7 Order Count",
    "week_8": "Week -8 Order Count",
    "week_9": "Week -9 Order Count",
    "week_10": "Week -10 Order Count",
    "week_11": "Week -11 Order Count",
    "avg_orders": "Average Weekly Orders",
    "std_orders": "Order Count Standard Deviation",
    "zero_weeks": "Number of Zero-Order Weeks",
    "recent_trend": "Recent Order Trend",
    "order_trend": "Overall Order Trend",
    "order_volatility": "Order Volatility",
    "recent_avg": "Recent Average Orders",
    "recent_std": "Recent Order Standard Deviation",
    "avg_order_total": "Average Order Total",
    "avg_discount": "Average Discount",
    "avg_loyalty": "Average Loyalty Earned",
    "total_spent": "Total Amount Spent",
    "avg_order_value": "Average Order Value",
    "discount_rate": "Discount Rate",
    "consecutive_zeros": "Consecutive Zero Orders",
    "max_consecutive_zeros": "Max Consecutive Zero Orders",
    "active_weeks": "Active Weeks Count",
    "activity_rate": "Activity Rate"
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
# Time series churn prediction endpoint
# --------------------------------------------------------------
def fetch_customer_time_series_data(engine):
    """Fetch customer time series data for the last 20 weeks"""
    query = """
    WITH last_20_weeks AS (
        SELECT DISTINCT week_end_date
        FROM mosaic_db.customer_tx_weekly
        WHERE WEEKDAY(week_end_date) = 5  -- Saturdays
        ORDER BY week_end_date DESC
        LIMIT 20
    ),
    all_customers AS (
        SELECT DISTINCT company_id, customer_id
        FROM mosaic_db.customer_tx_weekly
        WHERE week_end_date >= DATE_SUB(CURDATE(), INTERVAL 20 WEEK)
    )
    SELECT 
        c.company_id,
        c.customer_id,
        w.week_end_date,
        COALESCE(ctw.order_count, 0) AS order_count,
        COALESCE(ctw.order_total, 0) AS order_total,
        COALESCE(ctw.discount_total, 0) AS discount_total,
        COALESCE(ctw.loyalty_earned, 0) AS loyalty_earned
    FROM all_customers c
    CROSS JOIN last_20_weeks w
    LEFT JOIN mosaic_db.customer_tx_weekly ctw
        ON ctw.customer_id = c.customer_id
        AND ctw.company_id = c.company_id
        AND ctw.week_end_date = w.week_end_date
    ORDER BY c.customer_id, w.week_end_date DESC
    """
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

def create_time_series_features_for_prediction(df, lookback_weeks=12):
    """Create time series features for prediction (no future data leakage)"""
    df = df.sort_values(by=["customer_id", "week_end_date"])
    samples = []
    
    for customer_id, group in df.groupby("customer_id"):
        group = group.reset_index(drop=True)
        
        # Only create samples if customer has enough history
        if len(group) < lookback_weeks:
            continue
            
        # Take the most recent lookback_weeks for prediction
        history = group.iloc[:lookback_weeks]
        
        # Enhanced order count features
        order_counts = history["order_count"].tolist()
        order_totals = history["order_total"].tolist()
        discount_totals = history["discount_total"].tolist()
        loyalty_earned = history["loyalty_earned"].tolist()
        
        # Basic statistical features
        avg_orders = np.mean(order_counts)
        std_orders = np.std(order_counts)
        zero_weeks = sum(1 for x in order_counts if x == 0)
        
        # Trend features
        recent_trend = np.mean(order_counts[:4]) - np.mean(order_counts[-4:]) if len(order_counts) >= 8 else 0
        order_trend = np.polyfit(range(len(order_counts)), order_counts, 1)[0]
        
        # Volatility features
        order_volatility = np.std(order_counts[:6]) / (np.mean(order_counts[:6]) + 1e-8) if len(order_counts) >= 6 else 0
        
        # Recency features
        recent_avg = np.mean(order_counts[:4]) if len(order_counts) >= 4 else np.mean(order_counts)
        recent_std = np.std(order_counts[:4]) if len(order_counts) >= 4 else np.std(order_counts)
        
        # Monetary features
        avg_order_total = np.mean(order_totals)
        avg_discount = np.mean(discount_totals)
        avg_loyalty = np.mean(loyalty_earned)
        total_spent = np.sum(order_totals)
        
        # Customer value features
        avg_order_value = np.mean([t/c if c > 0 else 0 for t, c in zip(order_totals, order_counts)])
        discount_rate = np.sum(discount_totals) / (np.sum(order_totals) + 1e-8)
        
        # Behavioral patterns
        consecutive_zeros = 0
        max_consecutive_zeros = 0
        for count in order_counts:
            if count == 0:
                consecutive_zeros += 1
                max_consecutive_zeros = max(max_consecutive_zeros, consecutive_zeros)
            else:
                break
        
        # Frequency features
        active_weeks = sum(1 for x in order_counts if x > 0)
        activity_rate = active_weeks / len(order_counts)
        
        # Enhanced feature vector
        enhanced_features = order_counts + [
            avg_orders, std_orders, zero_weeks, recent_trend, order_trend,
            order_volatility, recent_avg, recent_std, avg_order_total, 
            avg_discount, avg_loyalty, total_spent, avg_order_value,
            discount_rate, consecutive_zeros, max_consecutive_zeros,
            active_weeks, activity_rate
        ]
        
        samples.append({
            "customer_id": customer_id,
            "features": enhanced_features
        })
    
    return pd.DataFrame(samples)

@app.post("/time-series-predict/")
def time_series_churn_predict():
    try:
        # 1. Fetch customer time series data
        engine = get_db_engine()
        df = fetch_customer_time_series_data(engine)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No customer time series data found")
        
        # 2. Create features for prediction
        feature_df = create_time_series_features_for_prediction(df)
        
        if feature_df.empty:
            raise HTTPException(status_code=404, detail="No customers with sufficient history for prediction")
        
        # 3. Prepare features for prediction
        X = pd.DataFrame(feature_df["features"].to_list())
        
        # 4. Scale features
        X_scaled = time_series_scaler.transform(X)
        
        # 5. Predict churn
        predictions = time_series_model.predict(X_scaled)
        prediction_probas = time_series_model.predict_proba(X_scaled)[:, 1]
        
        # 6. Filter only churned customers (prediction = 1)
        churned_indices = predictions == 1
        churned_customers = feature_df[churned_indices].copy()
        churned_customers['churn_probability'] = prediction_probas[churned_indices]
        
        # 7. Get feature importance for time series model
        if hasattr(time_series_model, 'feature_importances_'):
            importances = time_series_model.feature_importances_
        elif hasattr(time_series_model, 'coef_'):
            importances = abs(time_series_model.coef_[0])
        else:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(time_series_model, X_scaled, predictions, n_repeats=10, random_state=42)
            importances = result.importances_mean
        
        # 8. Get top 3 features for time series model
        feature_importance_pairs = list(zip(time_series_feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance_pairs[:3]
        
        # 9. Add top contributing factors to output
        churned_customers['top_factor_1'] = time_series_feature_importance_map.get(top_features[0][0], top_features[0][0])
        churned_customers['top_factor_2'] = time_series_feature_importance_map.get(top_features[1][0], top_features[1][0])
        churned_customers['top_factor_3'] = time_series_feature_importance_map.get(top_features[2][0], top_features[2][0])
        
        # 10. Select required columns
        output_columns = ['customer_id', 'churn_probability', 'top_factor_1', 'top_factor_2', 'top_factor_3']
        churned_customers = churned_customers[output_columns]
        
        # 11. Convert to CSV
        out = io.StringIO()
        churned_customers.to_csv(out, index=False)
        out.seek(0)
        
        return {
            "csv_data": out.getvalue(),
            "churn_count": len(churned_customers),
            "total_customers": len(feature_df),
            "churn_rate": len(churned_customers) / len(feature_df) if len(feature_df) > 0 else 0,
            "top_factors": [
                {"feature": time_series_feature_importance_map.get(feat, feat), "importance": float(imp)}
                for feat, imp in top_features
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sample-data/")
def get_sample_data(churn_count: int = 3, non_churn_count: int = 12):
    """Fetch sample data for customers with configurable churn and non-churn counts"""
    try:
        # Validate input parameters
        if churn_count < 0 or churn_count > 10:
            raise HTTPException(status_code=400, detail="churn_count must be between 0 and 10")
        if non_churn_count < 0 or non_churn_count > 40:
            raise HTTPException(status_code=400, detail="non_churn_count must be between 0 and 40")
        
        engine = get_db_engine()
        
        query = f"""
        WITH churn_customers AS (
            SELECT customer_id
            FROM (
                SELECT customer_id, MAX(week_end_date) AS last_week
                FROM mosaic_db.sample_data
                WHERE is_churn = 1
                GROUP BY customer_id
                ORDER BY last_week DESC
                LIMIT {churn_count}
            ) AS ordered_churn
        ),
        non_churn_customers AS (
            SELECT customer_id
            FROM (
                SELECT customer_id, MAX(week_end_date) AS last_week
                FROM mosaic_db.sample_data
                WHERE is_churn = 0
                GROUP BY customer_id
                ORDER BY last_week DESC
                LIMIT {non_churn_count}
            ) AS ordered_non_churn
        ),
        selected_customers AS (
            SELECT customer_id FROM churn_customers
            UNION ALL
            SELECT customer_id FROM non_churn_customers
        )
        SELECT 
            sd.customer_id,
            sd.week_end_date,
            sd.order_count,
            sd.order_total, 
            sd.discount_total,
            sd.loyalty_earned,
            sd.is_churn
        FROM mosaic_db.sample_data sd
        JOIN selected_customers sc ON sd.customer_id = sc.customer_id
        ORDER BY sd.is_churn, sd.customer_id, sd.week_end_date DESC
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No sample data found")
        
        # Convert to CSV for download
        out = io.StringIO()
        df.to_csv(out, index=False)
        out.seek(0)
        
        return {
            "csv_data": out.getvalue(),
            "total_records": len(df),
            "unique_customers": df['customer_id'].nunique(),
            "churned_customers": df[df['is_churn'] == 1]['customer_id'].nunique(),
            "non_churned_customers": df[df['is_churn'] == 0]['customer_id'].nunique(),
            "churn_count": churn_count,
            "non_churn_count": non_churn_count,
            "date_range": {
                "start_date": df['week_end_date'].min().strftime('%Y-%m-%d'),
                "end_date": df['week_end_date'].max().strftime('%Y-%m-%d')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class SampleDataRequest(BaseModel):
    data: list

@app.post("/llm-churn-predict/")
def llm_churn_predict(request: SampleDataRequest):
    """Predict churn using LLM with sample data from frontend"""
    try:
        # Convert the data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Ensure required columns exist
        required_columns = ['customer_id', 'week_end_date', 'order_count', 'order_total', 'discount_total', 'loyalty_earned']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        
        # Convert week_end_date to datetime
        df['week_end_date'] = pd.to_datetime(df['week_end_date'])
        
        # Prepare data for LLM analysis (same as churn_analysis.py)
        data = {}
        customer_ids = []
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id]
            customer_ids.append(str(customer_id))
            data[str(customer_id)] = {
                'total_orders': int(customer_data['order_count'].sum()),
                'total_revenue': float(customer_data['order_total'].sum()),
                'recent_orders': int(customer_data.tail(4)['order_count'].sum()),
                'weeks_active': len(customer_data)
            }
        
        # Use the same prompt as churn_analysis.py
        prompt = f"""
Analyze customer behavior over the most recent 20 continuous weeks, where each week ends on a Saturday. For any customer who does not have data for all 20 weeks, fill the missing weeks with zeros for all metrics. Based on this normalized 20-week view, perform a brief churn risk analysis.

Customer Data Summary:
{json.dumps(data, indent=2)}

Provide a concise summary (2-3 sentences) identifying:
1. How many customers are at high churn risk
2. Key patterns you observed
3. Main risk factors
4. List of customers at high churn risk
5. number of all provided customers

Keep the summary brief and actionable.
"""
        
        # Call OpenAI API
        try:
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a customer churn analysis expert. Provide concise, actionable summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            analysis_result = response.choices[0].message.content.strip()
            
            # Extract customer IDs from the analysis (simple extraction)
            # This is a basic extraction - you might want to improve this based on your needs
            high_risk_customers = []
            for customer_id in customer_ids:
                if customer_id in analysis_result:
                    high_risk_customers.append(customer_id)
            
            return {
                "analysis": analysis_result,
                "high_risk_customers": high_risk_customers,
                "total_customers_analyzed": len(customer_ids),
                "customer_data": data
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

