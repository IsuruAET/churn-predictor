from openai import OpenAI
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv(verbose=True)

# Configure OpenAI client
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def generate_churn_recommendations(top_factors: List[Dict], customer_data: Dict = None) -> str:
    """
    Generate personalized recommendations based on top contributing factors.
    
    Args:
        top_factors: List of dictionaries with 'feature' and 'importance' keys
        customer_data: Optional customer data for more personalized recommendations
    
    Returns:
        String with personalized recommendations
    """
    
    if not client.api_key:
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Build the prompt
    factors_text = "\n".join([f"- {factor['feature']} (Impact: {factor['importance']:.3f})" 
                             for factor in top_factors])
    
    if customer_data:
        customer_data_text = f"""
Customer context:
- Days since last order: {customer_data.get('days_since_last_order', 'N/A')}
- Average weekly order count: {customer_data.get('avg_order_count_weekly', 'N/A')}
- Average monthly order count: {customer_data.get('avg_order_count_monthly', 'N/A')}
- Average weekly order total: ${customer_data.get('avg_order_total_weekly', 'N/A')}
- Average monthly order total: ${customer_data.get('avg_order_total_monthly', 'N/A')}
"""
    else:
        customer_data_text = ""
    
    prompt = f"""
You are a customer retention expert for an e-commerce business. A customer has been identified as having a high risk of churning based on these top contributing factors:

{factors_text}

{customer_data_text}

Please provide 3-4 specific, actionable recommendations to prevent this customer from churning. Focus on:
1. Immediate actions (next 24-48 hours)
2. Medium-term strategies (next 1-2 weeks)
3. Long-term retention tactics

Make the recommendations practical and specific to the factors identified. Keep each recommendation concise but actionable.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a customer retention expert specializing in e-commerce. Provide practical, actionable advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}"

def generate_weekly_churn_recommendations(top_factors: List[str]) -> str:
    """
    Generate recommendations for weekly churn prediction based on top factors.
    
    Args:
        top_factors: List of top contributing factor names
    
    Returns:
        String with recommendations
    """
    
    if not client.api_key:
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    factors_text = "\n".join([f"- {factor}" for factor in top_factors])
    
    prompt = f"""
You are a customer retention expert. Based on analysis of customer behavior patterns, these are the top factors contributing to churn risk:

{factors_text}

Provide 3-4 strategic recommendations for reducing churn based on these patterns. Focus on:
1. Proactive customer engagement strategies
2. Targeted retention campaigns
3. Product/service improvements
4. Customer experience enhancements

Make recommendations specific to the identified behavioral patterns.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a customer retention expert specializing in e-commerce. Provide strategic, actionable advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}" 