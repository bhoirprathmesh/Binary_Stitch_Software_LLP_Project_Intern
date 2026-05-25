import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import os

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(__file__)
    model = joblib.load(os.path.join(current_dir, "best_broadcast_model.pkl"))
    meta = joblib.load(os.path.join(current_dir, "model_features.pkl"))
    return model, meta

def run():
    st.title("📺 IPL Broadcast Viewership Prediction")
    st.markdown("Predicts total_viewership_millions using cleaned, non-leaky features.")



    try:
        model, meta = load_model()
        st.sidebar.success("✅ Model loaded successfully")
        st.sidebar.write("Sklearn version:", sklearn.__version__)
    except Exception as e:
        st.sidebar.warning("⚠️ Cached model failed to load. Please re-train or prepare a new dataset to forecast.")
        model = None
        # Let the UI render instead of stopping

    
    # --------------------------------------------------
    # LOAD DATA DIRECTLY
    # --------------------------------------------------
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "ipl_broadcast_viewership.csv")
    
    df_hist = None
    if os.path.exists(csv_path):
        df_hist = pd.read_csv(csv_path)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.sort_values("date")
    
        st.sidebar.success(f"Loaded {len(df_hist)} rows")
        st.sidebar.metric("Avg Viewership",
                          f"{df_hist['total_viewership_millions'].mean():.2f} M")
    
    # --------------------------------------------------
    # DEFAULT LAG VALUES
    # --------------------------------------------------
    if df_hist is not None and not df_hist.empty:
        default_lag = df_hist['total_viewership_millions'].iloc[-1]
        default_roll = df_hist['total_viewership_millions'].tail(3).mean()
    else:
        default_lag = 6.5
        default_roll = 6.5
    
    # --------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------
    st.subheader("🎯 Predict Match Viewership")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_team = st.text_input("Home Team", "RCB")
        away_team = st.text_input("Away Team", "MI")
        broadcaster = st.text_input("Broadcaster", "Star Sports")
    
    with col2:
        tournament_phase = st.selectbox("Tournament Phase",
                                        ["League", "Playoff", "Final"])
        day_of_week = st.selectbox("Day of Week",
                                   ["Monday", "Tuesday", "Wednesday",
                                    "Thursday", "Friday", "Saturday", "Sunday"])
        time_slot = st.selectbox("Time Slot",
                                 ["Afternoon", "Evening", "Night"])
    
    with col3:
        city = st.text_input("City", "Mumbai")
        viewership_lag_1 = st.number_input(
            "Previous Match Viewership (M)",
            value=float(default_lag)
        )
        viewership_roll_3 = st.number_input(
            "Last 3 Match Avg (M)",
            value=float(default_roll)
        )
    
    # Match flags
    st.markdown("---")
    colA, colB, colC = st.columns(3)
    
    with colA:
        marquee = st.checkbox("Marquee Match")
    with colB:
        rivalry = st.checkbox("Rivalry Match")
    with colC:
        rain = st.checkbox("Rain Expected")
    
    # Auto features
    today = datetime.today()
    month = today.month
    is_weekend = 1 if today.weekday() >= 5 else 0
    
    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    if st.button("📈 Predict Viewership"):
        if model is None:
            st.error("❌ Prediction aborted: Cannot generate forecasts because the machine learning model is unresponsive or missing.")
        else:
            input_df = pd.DataFrame({
                'viewership_lag_1': [viewership_lag_1],
                'viewership_roll_3': [viewership_roll_3],
                'weather_rain': [int(rain)],
                'marquee_match': [int(marquee)],
                'rivalry_match': [int(rivalry)],
                'month': [month],
                'is_weekend': [is_weekend],
                'home_team': [home_team],
                'away_team': [away_team],
                'tournament_phase': [tournament_phase],
                'day_of_week': [day_of_week],
                'time_slot': [time_slot],
                'city': [city],
                'broadcaster': [broadcaster]
            })
    
        try:
            prediction = model.predict(input_df)[0]
    
            st.success("### 📊 Forecast Result")
            st.metric("Predicted Viewership",
                      f"{prediction:.2f} Million")
    
        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))
    
    # --------------------------------------------------
    # HISTORICAL PLOT
    # --------------------------------------------------
    if df_hist is not None:
        st.subheader("📈 Historical Trend")
    
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_hist,
                     x="date",
                     y="total_viewership_millions",
                     ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("---")
    st.caption("Sklearn 1.8 compatible • Clean model • No leakage")

if __name__ == "__main__":
    run()
