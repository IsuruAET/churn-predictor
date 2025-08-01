# app.py
import os
os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"      # disable telemetry

import streamlit as st
import requests, pandas as pd
import numpy as np
import io

API_URL = "http://localhost:8000/"

st.set_page_config(page_title="Mosaic Churn Predictor", page_icon="🛒")
st.title("🛒 Customer Churn Predictor")

# ─────────────────────────── Sample Data Display ──────────────────────────────
st.subheader("📋 Customer Weekly Order Summary for the Most Recent 20 Consecutive Weeks")
st.write("Configure and load sample dataset with custom churn and non-churn customer counts.")

# Customer count selection sliders
col1, col2 = st.columns(2)
with col1:
    churn_count = st.slider("Churned Customers", 0, 10, 3, help="Select number of churned customers (0-10)")
with col2:
    non_churn_count = st.slider("Non-Churned Customers", 0, 40, 12, help="Select number of non-churned customers (0-40)")

# Show total customer count
total_customers = churn_count + non_churn_count
st.info(f"📊 Total customers to load: {total_customers} ({churn_count} churned + {non_churn_count} non-churned)")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sample_data(churn_count: int, non_churn_count: int):
    """Load sample data from API with custom counts"""
    try:
        r = requests.get(API_URL + f"sample-data/?churn_count={churn_count}&non_churn_count={non_churn_count}")
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"API error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data button
if st.button("📊 Load Data", type="primary"):
    with st.spinner(f"Loading data for {total_customers} customers..."):
        response_data = load_sample_data(churn_count, non_churn_count)
        
        if response_data:
            csv_data = response_data.get("csv_data", "")
            total_records = response_data.get("total_records", 0)
            unique_customers = response_data.get("unique_customers", 0)
            churned_customers = response_data.get("churned_customers", 0)
            non_churned_customers = response_data.get("non_churned_customers", 0)
            actual_churn_count = response_data.get("churn_count", 0)
            actual_non_churn_count = response_data.get("non_churn_count", 0)
            date_range = response_data.get("date_range", {})
            
            # Read returned CSV
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Store the original dataframe and metadata in session state
            st.session_state.original_df = df
            st.session_state.total_records = total_records
            st.session_state.unique_customers = unique_customers
            st.session_state.churned_customers = churned_customers
            st.session_state.non_churned_customers = non_churned_customers
            st.session_state.date_range = date_range
            st.session_state.display_df = df
            st.session_state.is_shuffled = False

# Display data if available in session state
if 'original_df' in st.session_state:
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", st.session_state.total_records)
    with col2:
        st.metric("Unique Customers", st.session_state.unique_customers)
    with col3:
        st.metric("Churned Customers", st.session_state.churned_customers)
    with col4:
        st.metric("Non-Churned Customers", st.session_state.non_churned_customers)
    
    # Display date range
    if st.session_state.date_range:
        st.info(f"📅 Data covers period: {st.session_state.date_range['start_date']} to {st.session_state.date_range['end_date']}")
    
    # Display the datagrid
    st.subheader("📊 Customer Weekly Order Data")
    
    # Add shuffle and reset buttons in a row
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Shuffle Data Order", type="secondary", key="shuffle_btn"):
            # Shuffle the dataframe
            df_shuffled = st.session_state.original_df.sample(frac=1, random_state=None).reset_index(drop=True)
            st.session_state.display_df = df_shuffled
            st.session_state.is_shuffled = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset to Original Order", type="secondary", key="reset_btn"):
            # Reset to original order
            st.session_state.display_df = st.session_state.original_df
            st.session_state.is_shuffled = False
            st.rerun()
    
    # Show shuffle status
    if st.session_state.is_shuffled:
        st.info("📊 Data order has been shuffled randomly")
    else:
        st.info("📊 Data in original order (sorted by churn status, customer ID, week end date)")
    
    # Display the dataframe
    st.dataframe(st.session_state.display_df, use_container_width=True)

# ─────────────────────────── LLM Churn Prediction ──────────────────────────────
st.subheader("🤖 LLM-Based Churn Prediction")
st.write("AI-powered churn prediction using OpenAI GPT-4 analysis of customer behavior patterns.")

# Check if data is loaded
if 'display_df' not in st.session_state:
    st.warning("⚠️ Please load data first using the 'Load Data' button above.")
else:
    if st.button("🤖 Predict Churn with LLM", type="primary"):
        with st.spinner("Analyzing customer data with AI..."):
            try:
                # Get the display_df from session state
                display_df = st.session_state.display_df
                
                # Prepare data for API (remove is_churn column and convert to list of dicts)
                api_data = display_df.drop(columns=['is_churn']).to_dict('records')
                
                # Send data to LLM prediction endpoint
                r = requests.post(API_URL + "llm-churn-predict/", json={"data": api_data})
                
                if r.status_code != 200:
                    st.warning(f"API error {r.status_code}: {r.text}")
                else:
                    response_data = r.json()
                    analysis = response_data.get("analysis", "")
                    high_risk_customers = response_data.get("high_risk_customers", [])
                    total_customers = response_data.get("total_customers_analyzed", 0)
                    customer_data = response_data.get("customer_data", {})
                    
                    # Display analysis results
                    st.subheader("🔍 AI Analysis Results")
                    st.write(analysis)
                    
                    # Display high-risk customers
                    if high_risk_customers:
                        st.error(f"⚠️ AI identified {len(high_risk_customers)} high-risk customers")
                        
                        # Filter and display high-risk customers from the original data
                        high_risk_df = display_df[display_df['customer_id'].astype(str).isin(high_risk_customers)]
                        
                        if not high_risk_df.empty:
                            st.subheader("🚨 High-Risk Customers (AI Identified)")
                            st.dataframe(high_risk_df, use_container_width=True)
                            
                            # Download button for high-risk customers
                            csv_data = high_risk_df.to_csv(index=False)
                            st.download_button("Download high-risk customers list",
                                               csv_data.encode('utf-8'),
                                               file_name="llm_high_risk_customers.csv")
                    else:
                        st.success("✅ AI analysis found no high-risk customers")
                    
                    # Display customer summary statistics
                    st.subheader("📊 Customer Summary Statistics")
                    if customer_data:
                        summary_df = pd.DataFrame.from_dict(customer_data, orient='index')
                        summary_df.reset_index(inplace=True)
                        summary_df.rename(columns={'index': 'customer_id'}, inplace=True)
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers Analyzed", total_customers)
                    with col2:
                        st.metric("High-Risk Customers", len(high_risk_customers))
                    with col3:
                        risk_rate = len(high_risk_customers) / total_customers if total_customers > 0 else 0
                        st.metric("Risk Rate", f"{risk_rate:.1%}")
                        
            except Exception as e:
                st.error(f"Error during LLM prediction: {str(e)}")

# ─────────────────────────── Time Series Churn Prediction ──────────────────────────────
st.subheader("📈 ML-Based Time Series Churn Prediction")
st.write("Advanced churn prediction using time series analysis of customer behavior patterns over the last 12 weeks.")

if st.button("📊 Predict Churn with Time Series Model", type="primary"):
    with st.spinner("Analyzing customer time series data and predicting churn..."):
        r = requests.post(API_URL + "time-series-predict/")
        if r.status_code != 200:
            st.warning(f"API error {r.status_code}: {r.text}")
        else:
            response_data = r.json()
            csv_data = response_data.get("csv_data", "")
            churn_count = response_data.get("churn_count", 0)
            total_customers = response_data.get("total_customers", 0)
            churn_rate = response_data.get("churn_rate", 0)
            top_factors = response_data.get("top_factors", [])
            
            # Read returned CSV
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Customers", churn_count)
            with col2:
                st.metric("Total Customers", total_customers)
            with col3:
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            
            if churn_count > 0:
                st.error(f"⚠️ Found {churn_count} customers at high churn risk")
                
                # Display churn customers table
                st.subheader("🚨 High Risk Customers")
                st.dataframe(df)
                
                # Download button
                st.download_button("Download time series churn customers list",
                                   csv_data.encode('utf-8'),
                                   file_name="time_series_churn_customers.csv")
                
                # Display top contributing factors
                if top_factors:
                    st.subheader("🔍 Top Contributing Factors (Time Series Model)")
                    for i, factor in enumerate(top_factors, 1):
                        importance_pct = (factor['importance'] / sum(f['importance'] for f in top_factors)) * 100
                        st.write(f"**{i}.** {factor['feature']} ({importance_pct:.1f}% impact)")
                
                # Bin churn probabilities into 10 ranges (0.0 to 1.0 in steps of 0.1)
                bin_edges = np.arange(0, 1.1, 0.1)
                bin_labels = [f"{round(bin_edges[i], 1)}–{round(bin_edges[i+1], 1)}" for i in range(len(bin_edges)-1)]
                df['probability_bin'] = pd.cut(df['churn_probability'], bins=bin_edges, labels=bin_labels, include_lowest=True)

                # Count how many customers fall into each bin and filter out empty bins
                bin_counts = df['probability_bin'].value_counts().sort_index()
                bin_counts = bin_counts[bin_counts > 0]  # Show only bins with customers

                # Display as bar chart
                st.subheader("📊 Churn Probability Range Distribution")
                st.bar_chart(bin_counts)
            else:
                st.success("✅ No customers predicted to churn based on time series analysis")
