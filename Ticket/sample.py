import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit setup
# st.set_page_config(page_title="🎟️ IPL Ticket & Revenue Forecast Dashboard", layout="wide")

def run():
    st.title("🏏 IPL Ticket Price & Revenue Forecast Dashboard")

    import os
    # --- Load Data Directly ---
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "cricket_ticket_forecasting_dataset_updated.csv")

    if os.path.exists(csv_path):
        # --- Load and preview ---
        df = pd.read_csv(csv_path)
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # --- Preprocessing ---
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Encode categorical variables
        le_team = LabelEncoder()
        le_venue = LabelEncoder()
        le_day = LabelEncoder()
        le_time = LabelEncoder()

        for col in ['team1', 'team2', 'venue', 'day_of_week', 'time_of_day']:
            df[col] = df[col].astype(str)

        df['team1_enc'] = le_team.fit_transform(df['team1'])
        df['team2_enc'] = le_team.fit_transform(df['team2'])
        df['venue_enc'] = le_venue.fit_transform(df['venue'])
        df['day_enc'] = le_day.fit_transform(df['day_of_week'])
        df['time_enc'] = le_time.fit_transform(df['time_of_day'])

        # Replace missing values with 0 or mean
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # --- Features & Target ---
        features = [
            'team1_enc', 'team2_enc', 'venue_enc', 'team1_rank', 'team2_rank',
            'weather_temp', 'weather_rain', 'historical_avg_attendance',
            'ticket_price_min', 'ticket_price_max', 'dynamic_pricing_flag',
            'holiday_flag', 'ad_spend', 'social_media_mentions', 'search_trends_index',
            'promotion_flag', 'past_30d_avg_sales', 'expected_attendance',
            'venue_capacity', 'traffic_index', 'special_event_flag', 'day_enc', 'time_enc'
        ]
        target = 'ticket_price_avg'

        # Drop missing target rows
        df = df.dropna(subset=[target])

        X = df[features]
        y = df[target]

        # --- Train-Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Train Model ---
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # --- Evaluate Model ---
        r2 = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        # st.markdown(f"### ✅ Model Performance: R² = `{r2:.2f}`")

        # --- Prediction Section ---
        st.subheader("🎯 Predict Ticket Price, Tickets Sold & Revenue")

        col1, col2, col3 = st.columns(3)
        with col1:
            team1 = st.selectbox("Team 1", df['team1'].unique(), key="t1")
            team2 = st.selectbox("Team 2", df['team2'].unique(), key="t2")
            venue = st.selectbox("Venue", df['venue'].unique(), key="v")
            time_of_day = st.selectbox("Time of Day", df['time_of_day'].unique(), key="tod")
            day_of_week = st.selectbox("Day of Week", df['day_of_week'].unique(), key="dow")

        with col2:
            team1_rank = st.number_input("Team 1 Rank", min_value=1, max_value=10, value=4, key="t1r")
            team2_rank = st.number_input("Team 2 Rank", min_value=1, max_value=10, value=6, key="t2r")
            weather_temp = st.number_input("Temperature (°C)", min_value=10, max_value=45, value=33, key="wt")
            weather_rain = st.number_input("Rain (%)", min_value=0, max_value=100, value=2, key="wr")
            traffic_index = st.number_input("Traffic Index (1–10)", min_value=1, max_value=10, value=3, key="ti")

        with col3:
            ad_spend = st.number_input("Ad Spend (₹)", min_value=0, value=150000, key="ads")
            social_media_mentions = st.number_input("Social Media Mentions", min_value=0, value=20000, key="smm")
            search_trends_index = st.number_input("Search Trends Index", min_value=0, max_value=100, value=75, key="sti")
            past_30d_avg_sales = st.number_input("Past 30-Day Avg Sales", min_value=0, value=40000, key="p3s")
            venue_capacity = st.number_input("Venue Capacity", min_value=10000, max_value=100000, value=62134, key="vc")

        holiday_flag = st.checkbox("Holiday?", value=False, key="hf")
        promotion_flag = st.checkbox("Promotion Active?", value=True, key="pf")
        special_event_flag = st.checkbox("Special Event?", value=True, key="sef")
        dynamic_pricing_flag = st.checkbox("Dynamic Pricing?", value=True, key="dpf")

        if st.button("💰 Predict Ticket Metrics"):
            # Prepare input data
            input_data = pd.DataFrame({
                'team1_enc': [le_team.transform([team1])[0]],
                'team2_enc': [le_team.transform([team2])[0]],
                'venue_enc': [le_venue.transform([venue])[0]],
                'team1_rank': [team1_rank],
                'team2_rank': [team2_rank],
                'weather_temp': [weather_temp],
                'weather_rain': [weather_rain],
                'historical_avg_attendance': [df['historical_avg_attendance'].mean()],
                'ticket_price_min': [df['ticket_price_min'].mean()],
                'ticket_price_max': [df['ticket_price_max'].mean()],
                'dynamic_pricing_flag': [int(dynamic_pricing_flag)],
                'holiday_flag': [int(holiday_flag)],
                'ad_spend': [ad_spend],
                'social_media_mentions': [social_media_mentions],
                'search_trends_index': [search_trends_index],
                'promotion_flag': [int(promotion_flag)],
                'past_30d_avg_sales': [past_30d_avg_sales],
                'expected_attendance': [df['expected_attendance'].mean()],
                'venue_capacity': [venue_capacity],
                'traffic_index': [traffic_index],
                'special_event_flag': [int(special_event_flag)],
                'day_enc': [le_day.transform([day_of_week])[0]],
                'time_enc': [le_time.transform([time_of_day])[0]]
            })

            # Predict ticket price
            predicted_price = model.predict(input_data)[0]

            # Derived metrics
            base_demand_factor = (social_media_mentions / 20000) + (ad_spend / 1000000)
            tickets_sold = int(min(venue_capacity, (df['tickets_sold'].mean() * base_demand_factor)))
            total_revenue = predicted_price * tickets_sold
            capacity_utilization = (tickets_sold / venue_capacity) * 100

            # Display Results
            st.markdown("### 📈 Prediction Summary")
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("💵 Predicted Avg Ticket Price", f"₹{predicted_price:,.2f}")
            met2.metric("🎫 Predicted Tickets Sold", f"{tickets_sold:,}")
            met3.metric("💰 Predicted Total Revenue", f"₹{total_revenue:,.0f}")
            met4.metric("📊 Capacity Utilization", f"{capacity_utilization:.1f}%")

        # --- Visualizations ---
        st.subheader("📊 Data Insights & Trends")

        tab1, tab2, tab3 = st.tabs(["💹 Price & Revenue Trends", "🏟️ Team & Venue Insights", "🌦️ Marketing & Weather Impact"])

        with tab1:
            df_sorted = df.sort_values('date')
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df_sorted['date'], df_sorted['ticket_price_avg'], marker='o', label='Avg Price')
            ax.set_xlabel("Date")
            ax.set_ylabel("₹ Price")
            ax.set_title("Ticket Price Trend Over Time")
            ax.legend()
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(10,4))
            df_sorted['revenue'] = df_sorted['ticket_price_avg'] * df_sorted['tickets_sold']
            ax2.plot(df_sorted['date'], df_sorted['revenue'], color='green', marker='o', label='Revenue')
            ax2.set_title("Revenue Trend Over Time")
            ax2.set_ylabel("Total Revenue (₹)")
            st.pyplot(fig2)

        with tab2:
            colA, colB = st.columns(2)
            with colA:
                fig3, ax3 = plt.subplots()
                team_avg = df.groupby('team1')['ticket_price_avg'].mean().sort_values(ascending=False)
                sns.barplot(x=team_avg.values, y=team_avg.index, ax=ax3)
                ax3.set_title("Team-wise Avg Ticket Price")
                st.pyplot(fig3)
            with colB:
                fig4, ax4 = plt.subplots()
                venue_avg = df.groupby('venue')['ticket_price_avg'].mean().sort_values(ascending=False).head(10)
                sns.barplot(x=venue_avg.values, y=venue_avg.index, ax=ax4)
                ax4.set_title("Top 10 Venues by Avg Ticket Price")
                st.pyplot(fig4)

        with tab3:
            colC, colD = st.columns(2)
            with colC:
                fig5, ax5 = plt.subplots()
                sns.scatterplot(data=df, x='ad_spend', y='ticket_price_avg', hue='promotion_flag', ax=ax5)
                ax5.set_title("Ad Spend vs Ticket Price (Promotion Effect)")
                st.pyplot(fig5)
            with colD:
                fig6, ax6 = plt.subplots()
                sns.scatterplot(data=df, x='weather_temp', y='tickets_sold', hue='special_event_flag', ax=ax6)
                ax6.set_title("Weather & Events vs Tickets Sold")
                st.pyplot(fig6)

    else:
        st.error(f"❌ Could not find dataset at `{csv_path}`")

if __name__ == "__main__":
    run()
