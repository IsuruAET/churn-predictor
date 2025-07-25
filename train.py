# --------------------------------------------------------------
# Config and Imports
# --------------------------------------------------------------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import os
from sqlalchemy.exc import SQLAlchemyError
from db_config import get_db_engine

# Config
MODEL_PATH = 'models/churn_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
NUMERIC_COLS_PATH = 'models/numeric_columns.json'
RANDOM_STATE = 42

# Features and target
FEATURES = [
    'days_since_last_order',
    'avg_order_count_weekly',
    'avg_order_count_monthly',
    'avg_order_total_weekly',
    'avg_order_total_monthly',
]
TARGET = 'churned'  # Assumes a binary target column exists

# --------------------------------------------------------------
# Data Fetching
# --------------------------------------------------------------
def fetch_customer_data(engine):
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
        ) AS avg_order_total_monthly
    FROM customer_tx_summary ts;
    """
    try:
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------
# Main Training Script
# --------------------------------------------------------------
if __name__ == "__main__":
    try:
        engine = get_db_engine()
        df = fetch_customer_data(engine)
        if df.empty:
            print("No data fetched from database. Exiting.")
            exit(1)
        print(f"Fetched {len(df)} rows from database.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # --------------------------------------------------------------
    # Churn Labeling
    # --------------------------------------------------------------
    # Improved churn labeling: churned if no order in last 90 days OR low weekly order count
    df['churned'] = (
        (df['days_since_last_order'] > 90) |
        (df['avg_order_count_weekly'] < 0.1)
    ).astype(int)

    print(f"Churn rate: {df['churned'].mean():.2%}")

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------
    # Drop rows with missing values in features/target
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------------
    # Model Training
    # --------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print(f'ROC AUC: {roc_auc_score(y_test, y_proba):.4f}')

    # Cross-validation (optional, comment out if not needed)
    # cv_scores = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring='roc_auc', n_jobs=-1)
    # print(f'CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    # --------------------------------------------------------------
    # Save Model, Scaler, and Feature List
    # --------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(NUMERIC_COLS_PATH, 'w') as f:
        json.dump(FEATURES, f)

    print(f'Model saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')
    print(f'Feature list saved to {NUMERIC_COLS_PATH}') 