import pandas as pd

# Load deliveries.csv (from Kaggle IPL dataset)
df = pd.read_csv(r"C:\Users\Prathmesh\OneDrive\Documents\Desktop\Major\Merchandies\deliveries.csv")

# ---------------------------
# Batting Stats
# ---------------------------
batting = df.groupby("batsman").agg(
    matches_played=("match_id", "nunique"),
    balls_faced=("ball", "count"),
    runs_scored=("batsman_runs", "sum"),
    fours=("batsman_runs", lambda x: (x == 4).sum()),
    sixes=("batsman_runs", lambda x: (x == 6).sum())
).reset_index()

batting["strike_rate"] = batting["runs_scored"] / batting["balls_faced"] * 100

# ---------------------------
# Bowling Stats
# ---------------------------
bowling = df.groupby("bowler").agg(
    matches_bowled=("match_id", "nunique"),
    balls_bowled=("ball", "count"),
    runs_conceded=("total_runs", "sum"),
    wickets=("player_dismissed", lambda x: x.notna().sum())
).reset_index()

bowling["economy"] = bowling["runs_conceded"] / (bowling["balls_bowled"] / 6)

# ---------------------------
# Merge Batting & Bowling
# ---------------------------
player_stats = pd.merge(
    batting, bowling,
    left_on="batsman", right_on="bowler",
    how="outer"
)

# Clean up
player_stats.rename(columns={"batsman": "player"}, inplace=True)
player_stats.drop(columns=["bowler"], inplace=True)

# Fill NaNs with 0 for missing values
player_stats.fillna(0, inplace=True)

# Save as CSV
player_stats.to_csv("player_stats.csv", index=False)

print("player_stats.csv generated successfully!")
