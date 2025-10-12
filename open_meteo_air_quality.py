import requests
import pandas as pd
from datetime import datetime

# Karachi coordinates
LAT, LON = 24.8607, 67.0011

# Your time range (OpenWeather data: 2025-10-11 → 2025-09-21)
START_DATE = "2025-09-21"
END_DATE = "2025-10-11"

print(f"📡 Fetching Air Quality data from {START_DATE} → {END_DATE}")

# Open-Meteo Air Quality API endpoint
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Parameters
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "timezone": "Asia/Karachi"
}

# API call
response = requests.get(url, params=params)
data = response.json()

if "hourly" in data:
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    
    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={"time": "datetime"}, inplace=True)
    
    # Reverse the dataframe to match OpenWeather's order (latest → oldest)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Save to CSV
    df.to_csv("karachi_air_quality_20days_hourly.csv", index=False)
    
    print(f"✅ Air quality data saved → 'karachi_air_quality_20days_hourly.csv'")
    print(f"📊 Total records: {len(df)}")
else:
    print("⚠️ No hourly data found in API response.")
    print("Response sample:", data)
