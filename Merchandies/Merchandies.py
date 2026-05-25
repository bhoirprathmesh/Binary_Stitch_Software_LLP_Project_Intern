import pandas as pd
import numpy as np

# 1. Load datasets
matches = pd.read_csv(r"C:\Users\Prathmesh\OneDrive\Documents\Desktop\Major\Merchandies\matches.csv")
player_stats = pd.read_csv(r"C:\Users\Prathmesh\OneDrive\Documents\Desktop\Major\Merchandies\player_stats.csv")

# 2. Create match_id for consistency
matches.rename(columns={"id": "match_id"}, inplace=True)

# 3. Create team-level long format (each match gives 2 rows: team1 & team2)
teams_long = pd.melt(
    matches,
    id_vars=["match_id", "Season", "city", "date", "winner", "result"],
    value_vars=["team1", "team2"],
    var_name="team_type",
    value_name="team"
)

# 4. Mark if team won that match
teams_long["is_winner"] = (teams_long["team"] == teams_long["winner"]).astype(int)

# 5. Sort by team and date, then compute rolling wins in last 5 matches
teams_long = teams_long.sort_values(["team", "date"])
teams_long["team_wins_last_5"] = (
    teams_long.groupby("team")["is_winner"]
    .transform(lambda x: x.rolling(5, min_periods=1).sum())
)

# 6. Build player performance index (career-level, since no per-match stats)
player_stats["performance_index"] = (
    player_stats["runs_scored"] * 0.5 +
    player_stats["wickets"] * 20 -
    player_stats["economy"] * 2 +
    player_stats["strike_rate"] * 0.1
)

# For now, assign random performance index per team (since no player→team mapping given)
team_avg_perf = {team: np.random.uniform(20, 50) for team in teams_long["team"].unique()}
teams_long["performance_index"] = teams_long["team"].map(team_avg_perf)

# 7. Simulate ad spend & sentiment
np.random.seed(42)
teams_long["ad_campaign_spend"] = np.random.randint(10_000, 100_000, size=len(teams_long))
teams_long["sentiment_score"] = np.random.normal(loc=0.5, scale=0.1, size=len(teams_long)).clip(0,1)

# 8. Merchandise sales model
teams_long["historical_avg_merch_sales"] = np.random.randint(50_000, 300_000, size=len(teams_long))
teams_long["merch_sales"] = (
    teams_long["historical_avg_merch_sales"] * (1 + 0.2 * teams_long["sentiment_score"]) +
    0.1 * teams_long["ad_campaign_spend"] +
    0.3 * teams_long["performance_index"]
).astype(int)

# 9. Save dataset
keep_cols = [
    "match_id", "date", "team", "performance_index",
    "team_wins_last_5", "ad_campaign_spend", "sentiment_score",
    "historical_avg_merch_sales", "merch_sales"
]
teams_long[keep_cols].to_csv("cricket_merch_sales_dataset.csv", index=False)

print("cricket_merch_sales_dataset.csv generated successfully!")
