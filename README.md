# Pro Sports Market Business Analysis

**Advanced Sports Market Forecasting** — An end-to-end machine learning and analytics platform for IPL (Indian Premier League) revenue streams: broadcast viewership, ticket sales, sponsorship value, and merchandise demand.

Developed as part of the **Binary Stitch Software LLP** internship / major project.

---

## 1. Overview

This project delivers a unified **Sports Market Intelligence Hub** that helps franchises and organizers forecast key business metrics before and during a cricket season. It combines:

- **Four predictive modules**, each targeting a distinct revenue or engagement channel
- A **master Streamlit dashboard** (`master_dashboard.py`) with an executive summary and module navigation
- **Interactive Streamlit apps** per module for scenario-based predictions and visual insights
- A **static web front-end** (`webapp/`) for a polished marketing-style presentation layer
- **Jupyter notebooks** for exploratory analysis, feature engineering, and model training

The system is built around IPL-style match data (teams, venues, dates, viewership, pricing flags, sponsor categories, and fan sentiment) and uses supervised learning (primarily **Random Forest** and **Gradient Boosting** pipelines) to support data-driven decisions on pricing, marketing spend, inventory, and sponsorship bids.

---

## 2. Business Problem

Professional sports leagues generate revenue across multiple channels. Without reliable forecasts, organizations risk:

| Revenue stream | Pain point | What this project solves |
|----------------|------------|---------------------------|
| **Broadcast** | Uncertain TV/digital reach affects ad inventory pricing | Predict **total viewership (millions)** per match from teams, phase, time slot, rivalry flags, and historical lags |
| **Tickets** | Fixed pricing and poor demand planning reduce fill rates and revenue | Forecast **average ticket price**, **tickets sold**, and **total revenue** using team ranks, weather, promotions, dynamic pricing, and venue capacity |
| **Sponsorship** | Brands over- or under-bid without ROI visibility | Estimate **sponsor value (₹ Cr)** and **ROI multiple** from viewership, social buzz, spend, and category |
| **Merchandise** | Overstocking or stockouts hurt margins | Predict **merch sales** and revenue from team form, ad spend, sentiment, and sales history (lags/rolling averages) |

**Practical use cases** (from project scope):

- **Pricing optimization** — Simulate attendance and revenue if ticket price or dynamic pricing changes
- **Marketing allocation** — Estimate impact of higher ad spend or promotion flags on demand
- **Staffing forecast** — Use expected attendance and special-event flags for venue operations
- **Inventory & logistics** — Align merchandise stock with predicted sales after wins/campaigns
- **Sponsorship negotiation** — Compare sponsor spend vs. estimated realized value

---

## 3. Dataset Overview

| Dataset | Location | Approx. size | Target variable | Key inputs |
|---------|----------|--------------|-----------------|------------|
| **IPL broadcast viewership** | `Views/ipl_broadcast_viewership.csv` | ~10,000 matches | `total_viewership_millions` | Home/away team, tournament phase, day/time slot, city, broadcaster, marquee/rivalry/rain flags, TV/digital splits |
| **Cricket ticket forecasting** | `Ticket/cricket_ticket_forecasting_dataset_updated.csv` | ~757 matches | `ticket_price_avg` (also derives tickets sold & revenue) | Teams, venue, ranks, weather, min/max price, dynamic pricing, holiday/promotion/special-event flags, ad spend, social/search trends, capacity, attendance |
| **Sponsorship value** | `Sponser/sponsorship_value_cricket.csv` | ~10,000 records | `sponsor_value_est_crores_inr` | TV/digital viewership, social buzz, sentiment, brand recall, ad integration, on-screen time, sponsor category/format, spend, ROI |
| **Merchandise sales** | `Merchandies/cricket_merch_sales_dataset.csv` | ~1,500+ rows | `merch_sales` | Team, performance index, wins in last 5, ad campaign spend, sentiment, historical averages |

**Supporting / reference data** (Merchandise module):

- `Merchandies/matches.csv`, `deliveries.csv`, `player_stats.csv` — IPL ball-by-ball and player context for extended analysis

**Feature engineering patterns** used across modules:

- Date features: `month`, `day_of_week`, `is_weekend`
- **Lag features**: previous match viewership / sponsor value / merch sales
- **Rolling averages**: e.g. last 3 matches viewership or merch sales
- **Leakage control** (viewership): removed highly correlated fields such as `past_viewership_avg_millions` and `ad_inventory_price_lakhs` before training (see `Views/main.ipynb`)

**Ticket dataset flags** (binary 0/1):

- `dynamic_pricing_flag` — surge/flexible pricing applied
- `holiday_flag` — weekend or holiday
- `promotion_flag` — discount/offer campaign
- `special_event_flag` — finals, rivalries, high-profile matches

---

## 4. Tools and Technologies

| Category | Stack |
|----------|--------|
| **Language** | Python 3 |
| **ML / data** | pandas, NumPy, scikit-learn, joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboards** | Streamlit |
| **Analysis** | Jupyter Notebook |
| **Front-end** | HTML5, CSS3, JavaScript, [Chart.js](https://www.chartjs.org/), Lucide icons |
| **Models** | Random Forest Regressor, Gradient Boosting, ElasticNet (training scripts); time-series options (LSTM) noted in project docs for tickets |

**Dependencies** — install via:

```bash
pip install -r requirements.txt
```

Packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `requests`

---

## 5. Approach

### Architecture

```
Major_final2/
├── master_dashboard.py          # Unified Streamlit hub (Executive Summary + 4 modules)
├── requirements.txt
├── webapp/                      # Static UI (index.html, style.css, script.js)
├── Views/                       # Broadcast viewership
│   ├── app.py                   # Streamlit prediction UI
│   ├── main.ipynb               # EDA, leakage removal, model training
│   ├── ipl_broadcast_viewership.csv
│   └── best_broadcast_model.pkl # Generated after training (not always in repo)
├── Ticket/                      # Ticket price & revenue
│   ├── sample.py                # Streamlit dashboard
│   ├── forecast.ipynb           # Model experimentation
│   └── cricket_ticket_forecasting_dataset_updated.csv
├── Sponser/                     # Sponsorship value
│   ├── sponsorship_dashboard.py
│   ├── train_sponsorship_value.py
│   └── sponsorship_value_cricket.csv
└── Merchandies/                 # Merchandise sales
    ├── train_merch_models.py
    ├── merchandies_pred.ipynb
    └── cricket_merch_sales_dataset.csv
```

### Modeling workflow

1. **Data loading & cleaning** — Parse dates, handle missing values, sort chronologically
2. **Feature engineering** — Lags, rolling means, encodings for teams/venues/categories
3. **Train/test split** — Random split (tickets) or time-based holdout (merchandise, sponsorship)
4. **Pipeline** — `ColumnTransformer` + imputation + scaling/one-hot encoding + regressor
5. **Deployment** — Streamlit forms for user inputs → live prediction + charts

### Module-specific methods

| Module | Algorithm | Notes |
|--------|-----------|--------|
| **Viewership** | Preprocessing pipeline + tree models (RF / Gradient Boosting compared in notebook) | Best model saved as `best_broadcast_model.pkl`; app loads via `joblib` |
| **Tickets** | Random Forest (`n_estimators=100`) on encoded categoricals | Predicts price; derives tickets sold & revenue from demand heuristics |
| **Sponsorship** | Random Forest in sklearn `Pipeline` | Trains on dashboard load from CSV; outputs value in ₹ Cr and ROI multiple |
| **Merchandise** | Random Forest with time-aware 80/20 split | Uses `performance_index`, sentiment, ad spend, and merch lags |

### Executive dashboard

`master_dashboard.py` provides:

- KPI cards (viewership, ticket revenue, sponsorship ROI, merch growth — illustrative aggregates)
- Revenue distribution and trend charts
- Sidebar navigation into each specialized module

---

## 6. Outcome

Deliverables produced by the project:

1. **Unified analytics hub** — Single entry point for all four forecasting domains
2. **Match-level viewership forecasts** — Millions of viewers with historical trend plots (when model artifact is present)
3. **Ticket economics** — Predicted average price, tickets sold, total revenue, and capacity utilization %
4. **Sponsorship intelligence** — Estimated sponsor value, ROI multiple, spend vs. value scatter plots, profit distribution
5. **Merchandise planning** — Predicted units sold and revenue; trends and ad-spend impact charts
6. **Documented model quality** (from notebooks / training runs):
   - Viewership: cleaned models target realistic **R² ~ 0.80–0.90** (post leakage removal)
   - Tickets: Random Forest **R² ~ 0.92** reported in `forecast.ipynb`
   - Sponsorship & merch: on-the-fly training with holdout evaluation in dashboards

**Business impact**: Enables scenario testing (e.g. toggle promotion, dynamic pricing, marquee match) before committing budget to ads, staffing, inventory, or sponsor packages.

---

## 7. Conclusion

This project demonstrates how **multi-channel sports analytics** can be operationalized with Python ML pipelines and Streamlit dashboards. By treating broadcast, tickets, sponsorship, and merchandise as linked but separate prediction problems—and by applying consistent feature engineering (lags, calendar effects, and categorical encodings)—the platform gives stakeholders a practical toolkit for **revenue optimization and risk reduction** in a league like the IPL.

---
