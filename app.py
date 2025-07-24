# app.py
import os
os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"      # disable telemetry

import streamlit as st
import requests, pandas as pd

API_URL = "http://localhost:8000/"

st.set_page_config(page_title="Mosaic Churn Predictor", page_icon="🛒")
st.title("🛒 Customer Churn Predictor")
st.write("Enter a single customer **or** upload a CSV to score many customers.")

# ─────────────────────────── Single-customer form ──────────────────────────
with st.form("single_customer"):
    st.subheader("Manual input")
    days_since_last_order = st.number_input("Days Since Last Order", 0.0, 180.0, step=1.0)
    avg_order_count_weekly = st.number_input("Avg. Weekly Order Count", 0.0, 10.0, step=1.0)
    avg_order_count_monthly = st.number_input("Avg. Monthly Order Count", 0.0, 20.0, step=1.0)
    avg_order_total_weekly = st.number_input("Avg. Weekly Order Total ($)", 0.0, 5000.0, step=10.0)
    avg_order_total_monthly = st.number_input("Avg. Monthly Order Total ($)", 0.0, 20000.0, step=50.0)

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
                pred = r.json().get("churn_prediction")
                if pred == 1:
                    st.error("⚠️ High churn risk – suggest retention offer.")
                else:
                    st.success("✅ Low churn risk – customer likely to stay.")
        except Exception as e:
            st.warning(f"Request failed: {e}")

# ─────────────────────────── Batch CSV upload ──────────────────────────────
st.subheader("Batch scoring (CSV)")
csv = st.file_uploader("Upload CSV with the same 5 columns", type="csv")

if csv and st.button("Score batch"):
    df = pd.read_csv(csv)
    st.write("Preview:", df.head())

    preds = []
    progress = st.progress(0, text="Scoring customers...")
    for i, row in df.iterrows():
        r = requests.post(API_URL + "predict/", json=row.to_dict())
        preds.append(r.json().get("churn_prediction", None))
        progress.progress((i + 1) / len(df))
    df["churn_prediction"] = preds
    progress.empty()

    st.success("Batch scoring complete")
    st.dataframe(df)
    st.download_button("Download results as CSV",
                       df.to_csv(index=False).encode(),
                       file_name="churn_predictions.csv")
