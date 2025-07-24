# train_and_export_mysql.py
"""
Connect to an mosaic_db MySQL database, extract customer records,
train a churn-prediction model (5 features),
and export artifacts for Hugging Face Spaces.
"""

# --------------------------------------------------------------
# 1. Connect to your database, get the dataset
# --------------------------------------------------------------
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from pathlib import Path
import json, joblib

# 🔒 Replace with real credentials
username = "root"
password = "root"
host     = "localhost"
port     = "3306"
database = "mosaic_db"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
)

query = """
SELECT 
    ts.company_id,
    ts.customer_id,
    TIMESTAMPDIFF(DAY, ts.latest_tx_date, NOW()) AS days_since_last_order,
    
    IF(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) = 0, 0, 
       ts.order_count / CEILING(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) / 7)
    ) AS avg_order_count_weekly,
    
    IF(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) = 0, 0, 
       ts.order_count / CEILING(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) / 30)
    ) AS avg_order_count_monthly,
    
    IF(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) = 0, 0, 
       ts.order_total / CEILING(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) / 7)
    ) AS avg_order_total_weekly,
    
    IF(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) = 0, 0, 
       ts.order_total / CEILING(TIMESTAMPDIFF(DAY, ts.earliest_tx_date, ts.latest_tx_date) / 30)
    ) AS avg_order_total_monthly,
    
    CASE 
        WHEN TIMESTAMPDIFF(DAY, ts.latest_tx_date, NOW()) > 90 THEN 1
        ELSE 0
    END AS churned

FROM customer_tx_summary ts;
"""

df = pd.read_sql(query, engine)

# --------------------------------------------------------------
# 2. Pre-process
# --------------------------------------------------------------
y = df["churned"].astype(int)
X = df.drop(columns=["company_id", "customer_id", "churned"])   # 5 numeric features
numeric_cols = X.columns.tolist()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------------
# 3. Train model
# --------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_tr, y_tr)

print("Model report:\n", classification_report(y_te, model.predict(X_te)))

# --------------------------------------------------------------
# 4. Save artefacts
# --------------------------------------------------------------
Path("models").mkdir(exist_ok=True)

joblib.dump(model,  "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

with open("models/numeric_columns.json", "w") as f:
    json.dump(numeric_cols, f, indent=2)

print("✅ Artefacts saved to /models  (churn_model.pkl, scaler.pkl, numeric_columns.json)") 