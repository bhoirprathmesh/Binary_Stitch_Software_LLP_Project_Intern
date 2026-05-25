import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def build_model(df):
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
    
    # Feature Engineering
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lags
    df['sponsor_value_lag_1'] = df.groupby('sponsor_category')['sponsor_value_est_crores_inr'].shift(1)
    df['sponsor_value_roll_3'] = df.groupby('sponsor_category')['sponsor_value_est_crores_inr'].shift(1).rolling(3, min_periods=1).mean()
    df.fillna(method='bfill', inplace=True)
    
    # Check what features exist
    num_cols = [c for c in numeric_features if c in df.columns]
    cat_cols = [c for c in categorical_features if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df['sponsor_value_est_crores_inr']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    model = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model, df

def run():
    st.title("🤝 IPL Sponsorship Value Prediction Dashboard")
    
    # --------------------------------------------------
    # Load Data Directly
    # --------------------------------------------------
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "sponsorship_value_cricket.csv")
    
    if os.path.exists(csv_path):
        raw_df = pd.read_csv(csv_path)
        
        # Train model locally from uploaded data to ensure SKLearn sync
        with st.spinner("Training predictive algorithms on your data..."):
            model, df = build_model(raw_df)
        
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())
    
        # Default Lags
        default_val_lag = df['sponsor_value_est_crores_inr'].iloc[-1]
        default_val_roll = df['sponsor_value_est_crores_inr'].tail(5).mean()
    
        # --------------------------------------------------
        # Prediction Form
        # --------------------------------------------------
        st.subheader("🎯 Predict Sponsor Value")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            sponsor_category = st.selectbox("Sponsor Category", df['sponsor_category'].unique(), key="sc")
            if 'broadcaster' in df.columns:
                broadcaster = st.selectbox("Broadcaster", df['broadcaster'].unique(), key="bc")
            else:
                broadcaster = "Star Sports"
                
            if 'match_type' in df.columns:
                match_type = st.selectbox("Match Type", df['match_type'].unique(), key="mt")
            else:
                match_type = "T20"
    
        with c2:
            time_slot = st.selectbox("Time Slot", df.get('time_slot', ['Evening']).unique(), key="ts")
            tv_view = st.number_input("TV Viewership (M)", value=30.0, key="tv")
            digital_view = st.number_input("Digital Viewership (M)", value=25.0, key="dv")
    
        with c3:
            social_mentions = st.number_input("Social Mentions", value=150000, key="sm")
            sponsor_spend = st.number_input("Sponsor Spend (Cr)", value=2.0, key="ss")
            team_momentum = st.number_input("Team Momentum (1-10)", value=5, key="tm")
    
        # Match Flags
        colA, colB = st.columns(2)
        marquee = colA.checkbox("Marquee Match", value=True, key="mm")
        weekend = colB.checkbox("Weekend Match", value=False, key="wm")
    
        if st.button("🔮 Predict"):
            input_data = pd.DataFrame({
                'viewership_tv_millions': [tv_view],
                'viewership_digital_millions': [digital_view],
                'social_buzz_volume': [social_mentions],
                'sentiment_score': [0.5], # defaulting some values if missing from simple input form
                'brand_recall_score': [80.0],
                'ad_integration_score': [0.8],
                'on_screen_time_seconds': [1200],
                'sponsor_spend_crores_inr': [sponsor_spend],
                'roi_estimated': [3.5],
                'team_momentum_index': [team_momentum],
                'is_weekend': [int(weekend)],
                'marquee_match': [int(marquee)],
                'match_type': [match_type],
                'time_slot': [time_slot],
                'broadcaster': [broadcaster],
                'sponsor_category': [sponsor_category],
                'league': ['IPL'],
                'home_team': ['RCB'],
                'away_team': ['MI'],
                'city': ['Mumbai'],
                'venue': ['Wankhede'],
                'ad_format': ['Jersey'],
                'month': [4], # April
                
                # Computed lags
                'sponsor_value_lag_1': [default_val_lag],
                'sponsor_value_roll_3': [default_val_roll]
            })
    
            # Ensure columns match training
            try:
                predicted_value = model.predict(input_data)[0]
                roi_multiple = predicted_value / sponsor_spend if sponsor_spend > 0 else 0
        
                st.markdown("### 📊 Sponsorship Forecast")
                m1, m2, m3 = st.columns(3)
                m1.metric("💰 Estimated Sponsor Value", f"₹{predicted_value:.2f} Cr")
                m2.metric("📈 ROI Multiple", f"{roi_multiple:.2f}x")
                m3.metric("📺 Expected Engagement", f"{tv_view + digital_view:.2f} M Views")
            except Exception as e:
                st.error(f"Prediction missing features or failed: {e}")
    
        # --------------------------------------------------
        # Visuals
        # --------------------------------------------------
        st.subheader("📊 Profitability & Market Insights")
    
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(
            data=df,
            x='sponsor_spend_crores_inr',
            y='sponsor_value_est_crores_inr',
            hue='sponsor_category',
            ax=ax
        )
        ax.set_title("Sponsor Spend vs Realized Value (ROI mapping)")
        st.pyplot(fig)
        
        # Second Visual: Profit Distribution
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        df['profit'] = df['sponsor_value_est_crores_inr'] - df['sponsor_spend_crores_inr']
        sns.histplot(df['profit'], bins=20, kde=True, ax=ax2, color="purple")
        ax2.set_title("Distribution of Net Profits Across Sponsorships (Value - Spend)")
        st.pyplot(fig2)
    
    else:
        st.error(f"❌ Could not find dataset at `{csv_path}`")

if __name__ == "__main__":
    run()
