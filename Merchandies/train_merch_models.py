import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
# st.set_page_config(
#     page_title="🧢 IPL Merchandising Forecast Dashboard",
#     layout="wide"
# )

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def feature_engineer(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df = df.sort_values('date').reset_index(drop=True)

    # Date features
    df['day_of_week'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Team-based lags
    df['days_since_last_match'] = (
        df.groupby('team')['date'].diff().dt.days.fillna(0)
    )

    df['merch_sales_lag_1'] = df.groupby('team')['merch_sales'].shift(1)
    df['merch_sales_lag_2'] = df.groupby('team')['merch_sales'].shift(2)
    df['merch_sales_roll_3'] = (
        df.groupby('team')['merch_sales']
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df = df.dropna(subset=['merch_sales', 'merch_sales_lag_1'])

    return df


def build_preprocessor(numeric_features, categorical_features):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])

def run():
    st.title("🧢 IPL Merchandising Sales & Revenue Forecast Dashboard")

    import os
    # --------------------------------------------------
    # Load Data Directly
    # --------------------------------------------------
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "cricket_merch_sales_dataset.csv")

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Feature engineering
        df = feature_engineer(df)

        # Features
        numeric_features = [
            'performance_index',
            'team_wins_last_5',
            'ad_campaign_spend',
            'sentiment_score',
            'historical_avg_merch_sales',
            'merch_sales_lag_1',
            'merch_sales_lag_2',
            'merch_sales_roll_3',
            'days_since_last_match',
            'month',
            'is_weekend'
        ]

        categorical_features = ['team', 'day_of_week']

        numeric_features = [c for c in numeric_features if c in df.columns]
        categorical_features = [c for c in categorical_features if c in df.columns]

        X = df[numeric_features + categorical_features]
        y = df['merch_sales']

        # Train / Test Split (time-aware)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        preprocessor = build_preprocessor(numeric_features, categorical_features)

        model = Pipeline([
            ('pre', preprocessor),
            ('rf', RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)

        r2 = r2_score(y_test, model.predict(X_test))
        # st.markdown(f"### ✅ Model Performance (R²): `{r2:.2f}`")

        # --------------------------------------------------
        # Prediction Section
        # --------------------------------------------------
        st.subheader("🎯 Predict Merch Sales & Revenue")

        col1, col2, col3 = st.columns(3)

        with col1:
            team = st.selectbox("Team", df['team'].unique(), key="mtm")
            performance_index = st.number_input("Performance Index", 0.0, 100.0, 65.0, key="mpi")
            team_wins_last_5 = st.number_input("Wins in Last 5 Matches", 0, 5, 3, key="mw")

        with col2:
            ad_campaign_spend = st.number_input("Ad Campaign Spend (₹)", 0, 5_000_000, 250_000, key="mas")
            sentiment_score = st.slider("Fan Sentiment Score", -1.0, 1.0, 0.4, key="mss")
            historical_avg_merch_sales = st.number_input("Historical Avg Merch Sales", 0, 500000, 80000, key="mhs")

        with col3:
            avg_price = st.number_input("Avg Merch Item Price (₹)", 100, 5000, 799, key="map")
            last_sales = st.number_input("Last Match Merch Sales", 0, 500000, 75000, key="mls")

        if st.button("🧮 Predict Merch Performance"):

            input_df = pd.DataFrame({
                'performance_index': [performance_index],
                'team_wins_last_5': [team_wins_last_5],
                'ad_campaign_spend': [ad_campaign_spend],
                'sentiment_score': [sentiment_score],
                'historical_avg_merch_sales': [historical_avg_merch_sales],
                'merch_sales_lag_1': [last_sales],
                'merch_sales_lag_2': [last_sales * 0.9],
                'merch_sales_roll_3': [last_sales],
                'days_since_last_match': [5],
                'month': [pd.Timestamp.today().month],
                'is_weekend': [1],
                'team': [team],
                'day_of_week': [pd.Timestamp.today().weekday()]
            })

            predicted_sales = int(model.predict(input_df)[0])
            predicted_revenue = predicted_sales * avg_price

            st.markdown("### 📈 Prediction Summary")
            m1, m2, m3 = st.columns(3)

            m1.metric("🧢 Predicted Merch Sales", f"{predicted_sales:,}")
            m2.metric("💰 Predicted Revenue", f"₹{predicted_revenue:,.0f}")
            m3.metric("📦 Avg Item Price", f"₹{avg_price}")

        # --------------------------------------------------
        # Visualizations
        # --------------------------------------------------
        st.subheader("📊 Merchandising Insights")

        tab1, tab2 = st.tabs(["📈 Sales Trends", "📣 Marketing Impact"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df, x='date', y='merch_sales', ax=ax)
            ax.set_title("Merchandise Sales Trend Over Time")
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df,
                x='ad_campaign_spend',
                y='merch_sales',
                hue='team',
                ax=ax
            )
            ax.set_title("Ad Spend vs Merch Sales")
            st.pyplot(fig)

    else:
        st.error(f"❌ Could not find dataset at `{csv_path}`")

if __name__ == "__main__":
    run()
