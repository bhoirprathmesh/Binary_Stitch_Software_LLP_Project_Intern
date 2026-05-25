import pandas as pd
import requests
import datetime
import time

# 1. Load match data (Kaggle IPL dataset or Cricsheet CSV)
matches = pd.read_csv("matches.csv")   # Replace with your Cricsheet/Kaggle dataset

# Select useful columns
matches = matches[['id','date','team1','team2','venue','winner']]

# Convert date to datetime
matches['date'] = pd.to_datetime(matches['date'], dayfirst=True, errors='coerce')
# matches['date'] = pd.to_datetime(matches['date'])
matches['day_of_week'] = matches['date'].dt.day_name()

# 2. Add dummy team rank (if not available)
import numpy as np
team_ranks = {team: np.random.randint(1,9) for team in pd.concat([matches['team1'], matches['team2']]).unique()}
matches['team1_rank'] = matches['team1'].map(team_ranks)
matches['team2_rank'] = matches['team2'].map(team_ranks)

# 3. Weather API (OpenWeatherMap) - historical data
API_KEY = "bd9293b7cf10fc668cbb580e2214eda3"
def get_weather(lat, lon, date):
    """Fetch historical weather for a given location and date"""
    timestamp = int(time.mktime(date.timetuple()))
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url)
        data = r.json()
        temp = data['current']['temp']
        rain = data['current'].get('rain', {}).get('1h', 0)
        return temp, rain
    except:
        return None, None

# 4. Add stadium coordinates (lat, lon) manually for a few venues
venue_coords = {
    "Wankhede Stadium": (18.9388, 72.8258),
    "M. A. Chidambaram Stadium": (13.0625, 80.2794),
    "Eden Gardens": (22.5646, 88.3433),
}

matches['weather_temp'] = None
matches['weather_rain'] = None

for idx, row in matches.iterrows():
    venue = row['venue']
    if venue in venue_coords:
        lat, lon = venue_coords[venue]
        temp, rain = get_weather(lat, lon, row['date'])
        matches.at[idx, 'weather_temp'] = temp
        matches.at[idx, 'weather_rain'] = rain
    else:
        matches.at[idx, 'weather_temp'] = np.random.randint(20, 35)
        matches.at[idx, 'weather_rain'] = np.random.choice([0, 0, 1, 2]) # fallback

# 5. Attendance estimate (if not available in dataset)
matches['ticket_price_avg'] = np.random.randint(200, 1000, size=len(matches))
matches['historical_avg_attendance'] = np.random.randint(20000, 55000, size=len(matches))

# 6. Save merged dataset
matches.to_csv("cricket_ticket_forecasting_dataset.csv", index=False)
print("cricket_ticket_forecasting_dataset.csv generated!")
