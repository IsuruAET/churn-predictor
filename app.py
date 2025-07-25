# app.py
import os
os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"      # disable telemetry

import streamlit as st
import requests, pandas as pd
import io

API_URL = "http://localhost:8000/"

st.set_page_config(page_title="Mosaic Churn Predictor", page_icon="🛒")
st.title("🛒 Customer Churn Predictor")
st.write("Enter a single customer **or** upload a CSV to score many customers.")

# ─────────────────────────── Sidebar ──────────────────────────
with st.sidebar:
    st.header("📊 Model Features")
    
    with st.expander("Standard Churn Model Features", expanded=True):
        st.write("**Features used by churn_model.pkl:**")
        features_standard = [
            "days_since_last_order",
            "avg_order_count_weekly", 
            "avg_order_count_monthly",
            "avg_order_total_weekly",
            "avg_order_total_monthly"
        ]
        for i, feature in enumerate(features_standard, 1):
            st.write(f"{i}. `{feature}`")
    
    with st.expander("Weekly Churn Model Features", expanded=True):
        st.write("**Features used by churn_model_weekly.pkl:**")
        features_weekly = [
            "last12weeks_order_count",
            "last12weeks_order_total", 
            "last12weeks_discount_total",
            "last12weeks_loyalty_earned"
        ]
        for i, feature in enumerate(features_weekly, 1):
            st.write(f"{i}. `{feature}`")

# ─────────────────────────── Single-customer form ──────────────────────────
with st.form("single_customer"):
    st.subheader("Manual input")
    days_since_last_order = st.number_input("Days Since Last Order", 0, step=1, value=70)
    avg_order_count_weekly = st.number_input("Avg. Weekly Order Count", 0.0, step=1.0, value=10.0)
    avg_order_count_monthly = st.number_input("Avg. Monthly Order Count", 0.0, step=1.0, value=30.0)
    avg_order_total_weekly = st.number_input("Avg. Weekly Order Total ($)", 0.0, 5000.0, step=10.0, value=100.0)
    avg_order_total_monthly = st.number_input("Avg. Monthly Order Total ($)", 0.0, 20000.0, step=50.0, value=500.0)

    if st.form_submit_button("Predict ⏩"):
        payload = {
            "days_since_last_order": days_since_last_order,
            "avg_order_count_weekly": avg_order_count_weekly,
            "avg_order_count_monthly": avg_order_count_monthly,
            "avg_order_total_weekly": avg_order_total_weekly,
            "avg_order_total_monthly": avg_order_total_monthly
        }
        try:
            r = requests.post(API_URL + "predict/", json=payload, timeout=15)
            if r.status_code != 200:
                st.warning(f"API error {r.status_code}: {r.text}")
            else:
                response_data = r.json()
                pred = response_data.get("churn_prediction")
                top_factors = response_data.get("top_contributing_factors", [])
                
                if pred == 1:
                    st.error("⚠️ High churn risk – suggest retention offer.")

                    # Display top contributing factors
                    if top_factors:
                        st.subheader("🔍 Top Contributing Factors")
                        for i, factor in enumerate(top_factors, 1):
                            importance_pct = (factor['importance'] / sum(f['importance'] for f in top_factors)) * 100
                            st.write(f"**{i}.** {factor['feature']} ({importance_pct:.1f}% impact)")
                    
                    # Display recommendations if available
                    recommendations = response_data.get("recommendations", "")
                    if recommendations:
                        st.subheader("💡 AI-Generated Recommendations")
                        st.info(recommendations)
                else:
                    st.success("✅ Low churn risk – customer likely to stay.")
                
        except Exception as e:
            st.warning(f"Request failed: {e}")

# ─────────────────────────── Batch CSV upload ──────────────────────────────
st.subheader("Batch scoring (CSV)")
csv = st.file_uploader("Upload CSV with the same 5 columns", type="csv")

if csv and st.button("Score batch"):
    # Send CSV to backend batch endpoint
    with st.spinner("Uploading and scoring batch..."):
        files = {"file": (csv.name, csv, "text/csv")}
        r = requests.post(API_URL + "batch-predict/", files=files)
        if r.status_code != 200:
            st.warning(f"API error {r.status_code}: {r.text}")
        else:
            # Read returned CSV
            import io
            df = pd.read_csv(io.StringIO(r.text))
            st.success("Batch scoring complete")
            st.dataframe(df)
            st.download_button("Download results as CSV",
                               r.content,
                               file_name="churn_predictions.csv")

# ─────────────────────────── Next Week Churn Prediction ──────────────────────────────
st.subheader("Next Week Churn Prediction")
st.write("Click the button below to analyze all customers and predict next week churn.")

if st.button("🔮 Predict Next Week Churn", type="primary"):
    with st.spinner("Fetching customer data and predicting next week churn..."):
        r = requests.post(API_URL + "next-week-predict/")
        if r.status_code != 200:
            st.warning(f"API error {r.status_code}: {r.text}")
        else:
            response_data = r.json()
            csv_data = response_data.get("csv_data", "")
            recommendations = response_data.get("recommendations", "")
            churn_count = response_data.get("churn_count", 0)
            
            # Read returned CSV
            df = pd.read_csv(io.StringIO(csv_data))
            st.error(f"Found {churn_count} customers predicted to churn next week")
            
            st.dataframe(df)
            st.download_button("Download churn customers list",
                               csv_data.encode('utf-8'),
                               file_name="next_week_churn_customers.csv")
            
            # Display top contributing factors for the model
            if 'top_factor_1' in df.columns and 'top_factor_2' in df.columns:
                st.subheader("🔍 Top Contributing Factors for Churn")
                top_factor_1 = df['top_factor_1'].iloc[0] if len(df) > 0 else "N/A"
                top_factor_2 = df['top_factor_2'].iloc[0] if len(df) > 0 else "N/A"
                st.write(f"**1.** {top_factor_1}")
                st.write(f"**2.** {top_factor_2}")
            
            # Display AI recommendations
            if recommendations:
                st.subheader("💡 AI-Generated Strategic Recommendations")
                st.info(recommendations)
