import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import quote
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get API tokens and city
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_TOKEN = os.getenv("OPENWEATHER_TOKEN")
CITY = os.getenv("CITY", "Lahore")

if not AQICN_TOKEN:
    raise RuntimeError("❌ Missing AQICN_TOKEN in .env file")
if not OPENWEATHER_TOKEN:
    raise RuntimeError("❌ Missing OPENWEATHER_TOKEN in .env file")

# ================================
# 1. Fetch AQICN pollutant data
# ================================
def fetch_aqicn_city(city_name: str):
    BASE_URL = "https://api.waqi.info/feed/"
    encoded_city = quote(city_name)
    url = f"{BASE_URL}{encoded_city}/?token={AQICN_TOKEN}"

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"AQICN returned non-ok status: {data}")

    return data["data"]


def parse_aqicn(data):
    iaqi = data.get("iaqi", {})
    time_info = data.get("time", {})

    # Safely extract UTC time
    timestamp = (
        pd.to_datetime(time_info.get("utc"))
        if "utc" in time_info
        else pd.to_datetime(time_info.get("s"))  # fallback to local time
        if "s" in time_info
        else pd.Timestamp.utcnow()
    )

    row = {
        "timestamp_utc": timestamp,
        "aqi_api": data.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "station_name": data.get("city", {}).get("name"),
        "fetched_at": pd.Timestamp.utcnow()
    }
    return pd.DataFrame([row])


# ================================
# 2. Fetch OpenWeather data
# ================================
def fetch_openweather(city_name: str):
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": OPENWEATHER_TOKEN, "units": "metric"}
    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    row = {
        "timestamp_utc": pd.to_datetime(datetime.utcfromtimestamp(data["dt"])),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "pressure": data["main"]["pressure"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "wind_deg": data["wind"].get("deg"),
        "weather_desc": data["weather"][0]["description"]
    }
    return pd.DataFrame([row])

# ================================
# 3. Merge & Feature Engineering
# ================================
def engineer_features(df):
    df['hour'] = df['timestamp_utc'].dt.hour
    df['day'] = df['timestamp_utc'].dt.day
    df['month'] = df['timestamp_utc'].dt.month
    df['dow'] = df['timestamp_utc'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    # Derived feature placeholder (computed once multiple records exist)
    df['aqi_change_rate'] = 0.0
    return df


def compute_aqi_change_rate(existing_csv, new_df):
    if os.path.exists(existing_csv):
        prev = pd.read_csv(existing_csv, parse_dates=["timestamp_utc"])
        prev = pd.concat([prev, new_df]).sort_values("timestamp_utc")
        prev['aqi_change_rate'] = prev['aqi_api'].diff().fillna(0)
        return prev
    else:
        new_df['aqi_change_rate'] = 0
        return new_df

# ================================
# 4. Main Pipeline
# ================================
def main():
    print(f"🌍 Fetching AQI and Weather data for {CITY}...")

    # Fetch pollutant + weather data
    aqicn_data = fetch_aqicn_city(CITY)
    df_aqi = parse_aqicn(aqicn_data)
    df_weather = fetch_openweather(CITY)

    # Merge on timestamp
    df = pd.merge_asof(df_aqi.sort_values("timestamp_utc"),
                       df_weather.sort_values("timestamp_utc"),
                       on="timestamp_utc", direction="nearest")

    # Feature engineering
    df = engineer_features(df)

    # Save to local CSV and compute AQI change rate
    os.makedirs("data/processed", exist_ok=True)
    csv_path = f"data/processed/aqi_weather_{CITY.lower().replace(' ', '_')}.csv"
    df_final = compute_aqi_change_rate(csv_path, df)

    df_final.to_csv(csv_path, index=False)
    print(f"✅ Data saved to: {csv_path}")
    print(df_final.tail(3))

# ================================
if __name__ == "__main__":
    main()
