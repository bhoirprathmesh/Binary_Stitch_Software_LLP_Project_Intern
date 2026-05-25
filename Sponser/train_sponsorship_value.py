"""
train_sponsorship_value.py

Usage:
    python train_sponsorship_value.py sponsorship_data.csv

Output:
    - Trains multiple models
    - Evaluates on time-based holdout
    - Saves best model to best_sponsorship_model.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet


# --------------------------------------------------
# Load & Prepare Data
# --------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
def feature_engineer(df):
    df['day_num'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_num'].isin([5, 6]).astype(int)

    # Lag feature on sponsor value
    df['sponsor_value_lag_1'] = df['sponsor_value_est_crores_inr'].shift(1)
    df['sponsor_value_roll_3'] = (
        df['sponsor_value_est_crores_inr']
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    df = df.dropna(subset=['sponsor_value_lag_1'])
    return df


# --------------------------------------------------
# Main Training Logic
# --------------------------------------------------
def main(csv_path):
    df = load_data(csv_path)
    df = feature_engineer(df)

    target = 'sponsor_value_est_crores_inr'

    numeric_features = [
        'viewership_tv_millions',
        'viewership_digital_millions',
        'social_buzz_volume',
        'sentiment_score',
        'brand_recall_score',
        'ad_integration_score',
        'on_screen_time_seconds',
        'sponsor_spend_crores_inr',
        'roi_estimated',
        'sponsor_value_lag_1',
        'sponsor_value_roll_3',
        'month',
        'is_weekend'
    ]

    categorical_features = [
        'league',
        'home_team',
        'away_team',
        'city',
        'venue',
        'broadcaster',
        'time_slot',
        'sponsor_category',
        'ad_format'
    ]

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Time-aware split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Preprocessing
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])

    models = {
        "ElasticNet": ElasticNet(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    best_model = None
    best_rmse = float("inf")

    print("\nTraining Sponsorship Value Models...\n")

    for name, model in models.items():
        pipe = Pipeline([
            ('pre', preprocessor),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipe

    print(f"\n✅ Best Model RMSE: {best_rmse:.3f}")
    joblib.dump(best_model, "best_sponsorship_model.pkl")
    print("💾 Saved as best_sponsorship_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Sponsorship CSV file path")
    args = parser.parse_args()
    main(args.csv)
