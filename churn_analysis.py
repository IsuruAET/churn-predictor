import pandas as pd
import json
import openai
from db_config import get_db_engine
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv(verbose=True)


def fetch_customer_data():
    """Fetch customer order data for last 20 weeks"""
    engine = get_db_engine()
    query = """
    WITH churn_customers AS (
        SELECT customer_id
        FROM (
            SELECT customer_id, MAX(week_end_date) AS last_week
            FROM mosaic_db.sample_data
            WHERE is_churn = 1
            GROUP BY customer_id
            ORDER BY last_week DESC
            LIMIT 5
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
            LIMIT 15
        ) AS ordered_non_churn
    ),
    selected_customers AS (
        SELECT customer_id FROM churn_customers
        UNION ALL
        SELECT customer_id FROM non_churn_customers
    )
    SELECT 
        sd.customer_id,  -- qualified
        sd.week_end_date,
        sd.order_count,
        sd.order_total, 
        sd.discount_total,
        sd.loyalty_earned
    FROM mosaic_db.sample_data sd
    JOIN selected_customers sc ON sd.customer_id = sc.customer_id
    ORDER BY sd.is_churn ASC, sd.customer_id, sd.week_end_date;
    """
    
    try:
        df = pd.read_sql(query, engine)
        df['week_end_date'] = pd.to_datetime(df['week_end_date'])
        return df
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

def analyze_with_llm(df):
    """Send data to LLM for analysis"""
    # Convert DataFrame to simple format
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
    
    print(f"Sending {len(data)} customers to LLM for analysis")

    # identifying patterns such as declining activity, gaps in ordering, or low engagement, and summarize key churn indicators for each customer.
    
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
        
        return response.choices[0].message.content.strip()
                
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Error analyzing customer data"

def main():
    print("Fetching customer data...")
    df = fetch_customer_data()
    
    if df.empty:
        print("No data fetched. Exiting.")
        return
    
    print(f"Fetched data for {len(df['customer_id'].unique())} customers")
    
    print("Analyzing with LLM...")
    summary = analyze_with_llm(df)
    
    print(f"\n=== CHURN ANALYSIS SUMMARY ===")
    print(summary)

if __name__ == "__main__":
    main() 