# --------------------------------------------------------------
# Config and Imports
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import os
from sqlalchemy.exc import SQLAlchemyError
from db_config import get_db_engine

# Config
MODEL_PATH = 'models/churn_model_time_series.pkl'
SCALER_PATH = 'models/scaler_time_series.pkl'
FEATURE_NAMES_PATH = 'models/feature_names_time_series.pkl'
RANDOM_STATE = 42
BASE_DATE = "2025-03-25"

# --------------------------------------------------------------
# Data Fetching
# --------------------------------------------------------------
def fetch_customer_time_series_data(engine):
    """Fetch customer time series data using the provided query"""
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
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------
def create_time_series_features(df, lookback_weeks=12, churn_window=4, min_history_weeks=16):
    """Create enhanced features ensuring no future data leakage"""
    df = df.sort_values(by=["customer_id", "week_end_date"])
    samples = []
    
    # Get global time boundaries
    min_date = df["week_end_date"].min()
    max_date = df["week_end_date"].max()
    
    # Reserve last 20% of time for testing
    split_date = min_date + (max_date - min_date) * 0.8
    
    for customer_id, group in df.groupby("customer_id"):
        group = group.reset_index(drop=True)
        
        # Only create samples if customer has enough history
        if len(group) < min_history_weeks:
            continue
            
        for i in range(lookback_weeks, len(group) - churn_window + 1):
            history = group.iloc[i-lookback_weeks:i]
            future = group.iloc[i:i+churn_window]
            current_date = history.iloc[-1]["week_end_date"]
            
            # Skip if we don't have enough future data
            if len(future) < churn_window:
                continue
            
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
            recent_trend = np.mean(order_counts[-4:]) - np.mean(order_counts[:4])
            order_trend = np.polyfit(range(len(order_counts)), order_counts, 1)[0]
            
            # Volatility features
            order_volatility = np.std(order_counts[-6:]) / (np.mean(order_counts[-6:]) + 1e-8)
            
            # Recency features
            recent_avg = np.mean(order_counts[-4:])
            recent_std = np.std(order_counts[-4:])
            
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
            for count in reversed(order_counts):
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
            
            # Churn label (more sophisticated)
            future_orders = future["order_count"].sum()
            churn_label = int(future_orders == 0)
            
            # Determine if this sample is for training or testing
            is_training = current_date < split_date
            
            samples.append({
                "customer_id": customer_id,
                "current_date": current_date,
                "features": enhanced_features,
                "is_churn": churn_label,
                "is_training": is_training
            })
    
    return pd.DataFrame(samples)

# --------------------------------------------------------------
# Main Training Script
# --------------------------------------------------------------
if __name__ == "__main__":
    try:
        engine = get_db_engine()
        df = fetch_customer_time_series_data(engine)
        if df.empty:
            print("No data fetched from database. Exiting.")
            exit(1)
        print(f"Fetched {len(df)} rows from database.")
        print(f"Data date range: {df['week_end_date'].min()} to {df['week_end_date'].max()}")
        print(f"Unique customers: {df['customer_id'].nunique()}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Save raw database data
    df.to_csv('data/customer_order_weekly_data.csv', index=False)

    # Create features and labels
    labeled_df = create_time_series_features(df)
    labeled_df.to_csv('data/time_series_labeled_data.csv', index=False)
    
    print(f"Created {len(labeled_df)} labeled samples")

    # --------------------------------------------------------------
    # Time-Based Train/Test Split
    # --------------------------------------------------------------
    train_df = labeled_df[labeled_df["is_training"] == True]
    test_df = labeled_df[labeled_df["is_training"] == False]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Training churn rate: {train_df['is_churn'].mean():.3f}")
    print(f"Test churn rate: {test_df['is_churn'].mean():.3f}")

    # Prepare features
    X_train = pd.DataFrame(train_df["features"].to_list())
    y_train = train_df["is_churn"]
    
    # Handle case where test set is empty
    if len(test_df) > 0:
        X_test = pd.DataFrame(test_df["features"].to_list())
        y_test = test_df["is_churn"]
    else:
        # If no test data, use a small portion of training data for testing
        test_size = max(1, len(train_df) // 5)  # 20% for testing
        X_test = X_train.iloc[-test_size:]
        y_test = y_train.iloc[-test_size:]
        X_train = X_train.iloc[:-test_size]
        y_train = y_train.iloc[:-test_size]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------------
    # Model Training with Hyperparameter Tuning
    # --------------------------------------------------------------
    
    # Define models to try
    models = {
        'random_forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        'logistic_regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        )
    }
    
    # Define parameter grids
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10, 100],
            'class_weight': ['balanced']
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = None
    
    # Handle class imbalance with SMOTE
    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, len(y_train[y_train==1])-1))
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original training set: {len(y_train)} samples")
    print(f"Balanced training set: {len(y_train_balanced)} samples")
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    
    # Try each model with hyperparameter tuning
    for model_name, model in models.items():
        print(f"\nTuning {model_name}...")
        
        # Use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=cv, 
            scoring='f1_weighted',  # Use F1 for imbalanced data
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test_scaled)
        f1_score = grid_search.score(X_test_scaled, y_test)
        
        print(f"{model_name} - Best F1 Score: {f1_score:.3f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        if f1_score > best_score:
            best_score = f1_score
            best_model = grid_search.best_estimator_
            best_model_name = model_name
    
    print(f"\nBest model: {best_model_name} with F1 score: {best_score:.3f}")
    model = best_model

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    # Additional metrics for imbalanced data
    if len(np.unique(y_test)) > 1:
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"\n=== ROC AUC Score: {auc_score:.3f} ===")
        except:
            print("\n=== ROC AUC Score: Not available ===")
    
    # Class distribution analysis
    print(f"\n=== Class Distribution ===")
    print(f"Test set - Non-churn: {sum(y_test == 0)}, Churn: {sum(y_test == 1)}")
    print(f"Predicted - Non-churn: {sum(y_pred == 0)}, Churn: {sum(y_pred == 1)}")

    # Feature importance
    feature_names = [f"week_{i}" for i in range(12)] + [
        "avg_orders", "std_orders", "zero_weeks", "recent_trend", "order_trend",
        "order_volatility", "recent_avg", "recent_std", "avg_order_total", 
        "avg_discount", "avg_loyalty", "total_spent", "avg_order_value",
        "discount_rate", "consecutive_zeros", "max_consecutive_zeros",
        "active_weeks", "activity_rate"
    ]
    
    # Handle different model types for feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For logistic regression, use absolute coefficients
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        print("Model doesn't support feature importance")
        importance_df = pd.DataFrame()

    print("\n=== Top 10 Feature Importances ===")
    print(importance_df.head(10))

    # --------------------------------------------------------------
    # Save Model, Scaler, and Feature Names
    # --------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print(f'\nModel saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')
    print(f'Feature names saved to {FEATURE_NAMES_PATH}')
    print("\nTraining completed successfully!")
