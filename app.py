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
                    
                    # Display simple comparison: Actual vs LLM Predicted Churn
                    st.subheader("🔍 Actual vs LLM Predicted Churn Comparison")
                    
                    # Get actual churn customers from the data
                    actual_churn_customers = display_df[display_df['is_churn'] == 1]['customer_id'].unique().tolist()
                    actual_churn_customers = [str(cid) for cid in actual_churn_customers]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Actual Churn Customers:**")
                        if actual_churn_customers:
                            for customer_id in actual_churn_customers:
                                st.write(f"• Customer {customer_id}")
                        else:
                            st.write("None")
                        st.metric("Actual Churn Count", len(actual_churn_customers))
                    
                    with col2:
                        st.write("**🤖 LLM Predicted Churn Customers:**")
                        if high_risk_customers:
                            for customer_id in high_risk_customers:
                                st.write(f"• Customer {customer_id}")
                        else:
                            st.write("None")
                        st.metric("Predicted Churn Count", len(high_risk_customers))
                    
                    # Show the difference
                    st.subheader("📈 Comparison Summary")
                    
                    # Calculate overlap
                    actual_set = set(actual_churn_customers)
                    predicted_set = set(high_risk_customers)
                    
                    correctly_identified = len(actual_set.intersection(predicted_set))
                    missed_churners = len(actual_set - predicted_set)
                    false_alarms = len(predicted_set - actual_set)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Correctly Identified", correctly_identified)
                    with col2:
                        st.metric("❌ Missed Churners", missed_churners)
                    with col3:
                        st.metric("⚠️ False Alarms", false_alarms)
                    
                    # Show detailed breakdown
                    if correctly_identified > 0:
                        st.success(f"**Correctly Identified:** {', '.join(actual_set.intersection(predicted_set))}")
                    
                    if missed_churners > 0:
                        st.error(f"**Missed Churners:** {', '.join(actual_set - predicted_set)}")
                    
                    if false_alarms > 0:
                        st.warning(f"**False Alarms:** {', '.join(predicted_set - actual_set)}")
                        
            except Exception as e:
                st.error(f"Error during LLM prediction: {str(e)}")

# ─────────────────────────── ML Churn Prediction ──────────────────────────────
st.subheader("🤖 ML-Based Churn Prediction")
st.write("ML-powered churn prediction using time series analysis of customer behavior patterns.")

# Customer count selection for ML prediction
col1, col2 = st.columns(2)
with col1:
    ml_churn_count = st.slider("ML Churned Customers", 0, 10, 3, help="Select number of churned customers for ML prediction (0-10)")
with col2:
    ml_non_churn_count = st.slider("ML Non-Churned Customers", 0, 40, 12, help="Select number of non-churned customers for ML prediction (0-40)")

# Show total customer count for ML
ml_total_customers = ml_churn_count + ml_non_churn_count
st.info(f"📊 Total customers for ML prediction: {ml_total_customers} ({ml_churn_count} churned + {ml_non_churn_count} non-churned)")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_ml_raw_data(churn_count: int, non_churn_count: int):
    """Load raw ML data from API with custom counts for viewing"""
    try:
        r = requests.get(API_URL + f"ml-raw-data/?churn_count={churn_count}&non_churn_count={non_churn_count}")
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"API error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        st.error(f"Error loading ML raw data: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_ml_data(churn_count: int, non_churn_count: int):
    """Load ML prediction data from API with custom counts"""
    try:
        r = requests.post(API_URL + f"ml-churn-predict/?churn_count={churn_count}&non_churn_count={non_churn_count}")
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"API error {r.status_code}: {r.text}")
            return None
    except Exception as e:
        st.error(f"Error loading ML data: {str(e)}")
        return None

# Load ML raw data button
if st.button("📊 Load ML Data", type="secondary"):
    with st.spinner(f"Loading raw data for {ml_total_customers} customers..."):
        response_data = load_ml_raw_data(ml_churn_count, ml_non_churn_count)
        
        if response_data:
            csv_data = response_data.get("csv_data", "")
            total_records = response_data.get("total_records", 0)
            unique_customers = response_data.get("unique_customers", 0)
            churned_customers = response_data.get("churned_customers", 0)
            non_churned_customers = response_data.get("non_churned_customers", 0)
            date_range = response_data.get("date_range", {})
            
            # Read returned CSV
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Store the ML dataframe and metadata in session state
            st.session_state.ml_raw_df = df
            st.session_state.ml_total_records = total_records
            st.session_state.ml_unique_customers = unique_customers
            st.session_state.ml_churned_customers = churned_customers
            st.session_state.ml_non_churned_customers = non_churned_customers
            st.session_state.ml_date_range = date_range

# Display ML raw data if available in session state
if 'ml_raw_df' in st.session_state:
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", st.session_state.ml_total_records)
    with col2:
        st.metric("Unique Customers", st.session_state.ml_unique_customers)
    with col3:
        st.metric("Churned Customers", st.session_state.ml_churned_customers)
    with col4:
        st.metric("Non-Churned Customers", st.session_state.ml_non_churned_customers)
    
    # Display date range
    if st.session_state.ml_date_range:
        st.info(f"📅 ML Data covers period: {st.session_state.ml_date_range['start_date']} to {st.session_state.ml_date_range['end_date']}")
    
    # Display the datagrid
    st.subheader("📊 ML Customer Weekly Order Data")
    st.dataframe(st.session_state.ml_raw_df, use_container_width=True)

# Load ML data button - only show if ML data is loaded
if 'ml_raw_df' in st.session_state:
    if st.button("🤖 Predict Churn with ML", type="primary"):
        with st.spinner(f"Analyzing {ml_total_customers} customers with ML model..."):
            try:
                response_data = load_ml_data(ml_churn_count, ml_non_churn_count)
                
                if response_data:
                    csv_data = response_data.get("csv_data", "")
                    churn_count = response_data.get("churn_count", 0)
                    total_customers = response_data.get("total_customers", 0)
                    churn_rate = response_data.get("churn_rate", 0)
                    top_factors = response_data.get("top_factors", [])
                    input_churn_count = response_data.get("input_churn_count", 0)
                    input_non_churn_count = response_data.get("input_non_churn_count", 0)
                    
                    # Read returned CSV
                    df = pd.read_csv(io.StringIO(csv_data))
                    
                    # Store ML results in session state for comparison
                    st.session_state.ml_results = {
                        'predicted_churn_customers': df['customer_id'].tolist() if not df.empty else [],
                        'churn_count': churn_count,
                        'total_customers': total_customers,
                        'churn_rate': churn_rate,
                        'top_factors': top_factors,
                        'input_churn_count': input_churn_count,
                        'input_non_churn_count': input_non_churn_count,
                        'csv_data': csv_data
                    }
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Predicted Churn", churn_count)
                    with col2:
                        st.metric("Total Customers", total_customers)
                    with col3:
                        st.metric("Churn Rate", f"{churn_rate:.1%}")
                    with col4:
                        st.metric("Input Customers", f"{input_churn_count + input_non_churn_count}")
                    
                    if churn_count > 0:
                        st.error(f"⚠️ Found {churn_count} customers at high churn risk")
                        
                        # Display churn customers table
                        st.subheader("🚨 ML Predicted High Risk Customers")
                        st.dataframe(df)
                        
                        # Download button
                        st.download_button("Download ML churn customers list",
                                           csv_data.encode('utf-8'),
                                           file_name="ml_churn_customers.csv")
                        
                        # Display top contributing factors
                        if top_factors:
                            st.subheader("🔍 Top Contributing Factors (ML Model)")
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
                        st.subheader("📊 ML Churn Probability Range Distribution")
                        st.bar_chart(bin_counts)
                    else:
                        st.success("✅ No customers predicted to churn based on ML analysis")
                        
            except Exception as e:
                st.error(f"Error during ML prediction: {str(e)}")
else:
    st.warning("⚠️ Please load ML data first using the 'Load ML Data' button above to enable ML prediction.")

# Display ML comparison if both ML raw data and ML results are available
if 'ml_raw_df' in st.session_state and 'ml_results' in st.session_state:
    st.subheader("🔍 Actual vs ML Predicted Churn Comparison")
    
    # Get actual churn customers from the ML raw data
    actual_churn_customers = st.session_state.ml_raw_df[st.session_state.ml_raw_df['is_churn'] == 1]['customer_id'].unique().tolist()
    actual_churn_customers = [str(cid) for cid in actual_churn_customers]
    
    # Get ML predicted churn customers
    ml_predicted_customers = st.session_state.ml_results['predicted_churn_customers']
    ml_predicted_customers = [str(cid) for cid in ml_predicted_customers]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Actual Churn Customers:**")
        if actual_churn_customers:
            for customer_id in actual_churn_customers:
                st.write(f"• Customer {customer_id}")
        else:
            st.write("None")
        st.metric("Actual Churn Count", len(actual_churn_customers))
    
    with col2:
        st.write("**🤖 ML Predicted Churn Customers:**")
        if ml_predicted_customers:
            for customer_id in ml_predicted_customers:
                st.write(f"• Customer {customer_id}")
        else:
            st.write("None")
        st.metric("ML Predicted Churn Count", len(ml_predicted_customers))
    
    # Show the difference
    st.subheader("📈 ML Comparison Summary")
    
    # Calculate overlap
    actual_set = set(actual_churn_customers)
    predicted_set = set(ml_predicted_customers)
    
    correctly_identified = len(actual_set.intersection(predicted_set))
    missed_churners = len(actual_set - predicted_set)
    false_alarms = len(predicted_set - actual_set)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Correctly Identified", correctly_identified)
    with col2:
        st.metric("❌ Missed Churners", missed_churners)
    with col3:
        st.metric("⚠️ False Alarms", false_alarms)
    
    # Show detailed breakdown
    if correctly_identified > 0:
        st.success(f"**Correctly Identified:** {', '.join(actual_set.intersection(predicted_set))}")
    
    if missed_churners > 0:
        st.error(f"**Missed Churners:** {', '.join(actual_set - predicted_set)}")
    
    if false_alarms > 0:
        st.warning(f"**False Alarms:** {', '.join(predicted_set - actual_set)}")

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
