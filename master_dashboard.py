import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Pro Sports Market Business Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS for Premium Look
# -------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #E50914;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #B3B3B3;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/5328/5328087.png", width=100) # Placeholder for logo
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["📊 Executive Summary", "📺 Broadcast Viewership", "🎟️ Ticket Revenue", "🤝 Sponsorship Value", "👕 Merchandise Sales"]
)

st.sidebar.markdown("---")
st.sidebar.info("Developed for Advanced Sports Market Forecasting.")

# -------------------------
# Mock Data Generators for Dashboard Overview
# -------------------------
@st.cache_data
def get_overview_data():
    dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
    return pd.DataFrame({
        'Date': dates,
        'Viewership (Millions)': np.random.uniform(20, 100, 12).cumsum() + 50,
        'Revenue (Cr)': np.random.uniform(5, 30, 12).cumsum() + 10,
        'Merch Sales (Lakhs)': np.random.uniform(10, 50, 12).cumsum() + 20
    })

df_trends = get_overview_data()

# -------------------------
# 1. Executive Summary
# -------------------------
if menu == "📊 Executive Summary":
    st.markdown('<div class="main-header">Sports Market Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive AI-driven business analytics for franchise valuation and revenue optimization.</div>', unsafe_allow_html=True)
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Viewership Forecast</div>
            <div class="metric-value">1.4B+</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #00C9FF;">
            <div class="metric-label">Projected Ticket Rev</div>
            <div class="metric-value">₹320 Cr</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #00B4DB;">
            <div class="metric-label">Avg Sponsorship ROI</div>
            <div class="metric-value">2.4x</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #F8B500;">
            <div class="metric-label">Merchandise Growth</div>
            <div class="metric-value">+18.5%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📈 Annual Revenue & Viewership Trends")
        st.line_chart(df_trends.set_index('Date')[['Viewership (Millions)', 'Revenue (Cr)']])
        
    with col_chart2:
        st.subheader("🎯 Revenue Distribution")
        # Creating a simple pie chart using matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0.0) # Transparent background
        labels = 'Broadcast', 'Sponsorship', 'Tickets', 'Merchandise'
        sizes = [45, 30, 15, 10]
        colors = ['#FF416C', '#00C9FF', '#F8B500', '#1E9600']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color':"w"})
        ax.axis('equal') 
        st.pyplot(fig)
        
    st.markdown("### 🚀 Module Quick Access")
    st.info("""
    Use the sidebar to navigate to specific predictive models:
    * **Broadcast Viewership**: Predict Match-day TV/Digital engagement.
    * **Ticket Revenue**: Forecast stadium occupancy and optimal pricing.
    * **Sponsorship Value**: Calculate Brand exposure and sponsorship bid values.
    * **Merchandise Sales**: Estimate fan gear sales based on team performance.
    """)

# -------------------------
# 2. Viewership Module
# -------------------------
elif menu == "📺 Broadcast Viewership":
    import Views.app
    Views.app.run()

# -------------------------
# 3. Ticket Revenue Module
# -------------------------
elif menu == "🎟️ Ticket Revenue":
    import Ticket.sample
    Ticket.sample.run()

# -------------------------
# 4. Sponsorship Module
# -------------------------
elif menu == "🤝 Sponsorship Value":
    import Sponser.sponsorship_dashboard
    Sponser.sponsorship_dashboard.run()

# -------------------------
# 5. Merchandise Module
# -------------------------
elif menu == "👕 Merchandise Sales":
    import Merchandies.train_merch_models
    Merchandies.train_merch_models.run()

