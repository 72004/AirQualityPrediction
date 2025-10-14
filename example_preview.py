import requests
import pandas as pd
import datetime
import pytz  # <-- added for timezone support

# ----------------------------------------------------------
# 🌤️ Fetch latest weather data
# ----------------------------------------------------------
def fetch_weather(api_key, city="Karachi"):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    data = requests.get(url, params=params).json()

    weather_data = {
        "timestamp_utc": datetime.datetime.utcnow(),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "wind_deg": data["wind"]["deg"],
        "clouds": data["clouds"]["all"],
        "weather": data["weather"][0]["main"],
        "weather_desc": data["weather"][0]["description"],
    }
    return pd.DataFrame([weather_data])

# ----------------------------------------------------------
# 🌫️ Fetch latest air quality data
# ----------------------------------------------------------
def fetch_aqi(lat=24.8607, lon=67.0011):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide",
                   "nitrogen_dioxide", "sulphur_dioxide",
                   "ozone", "us_aqi"],
        "timezone": "UTC"
    }

    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data["hourly"])
    latest = df.iloc[-1]

    aqi_data = {
        "timestamp_utc": pd.to_datetime(latest["time"]),
        "pm10": latest["pm10"],
        "pm25": latest["pm2_5"],
        "carbon_monoxide": latest["carbon_monoxide"],
        "nitrogen_dioxide": latest["nitrogen_dioxide"],
        "sulphur_dioxide": latest["sulphur_dioxide"],
        "ozone": latest["ozone"],
        "aqi_api": latest["us_aqi"],
        "station_name": "Karachi",
    }
    return pd.DataFrame([aqi_data])

# ----------------------------------------------------------
# 🔍 Display merged result (matching 35-column structure)
# ----------------------------------------------------------
def preview_merge():
    CITY = "Karachi"
    OPENWEATHER_API_KEY = "6002ac178c1ae72e8625cd919f36f421"  # replace this

    df_weather = fetch_weather(OPENWEATHER_API_KEY, CITY)
    df_aqi = fetch_aqi()

    merged = pd.merge_asof(
        df_weather.sort_values("timestamp_utc"),
        df_aqi.sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta("1h")
    )

    # 🧮 Add engineered & placeholder columns for schema consistency
    merged["temp_C"] = merged["temp"]
    merged["feels_like_C"] = merged["feels_like"]
    merged["humidity_%"] = merged["humidity"]
    merged["pressure_hPa"] = merged["pressure"]
    merged["wind_speed_m/s"] = merged["wind_speed"]
    merged["clouds_%"] = merged["clouds"]

    merged["hour"] = merged["timestamp_utc"].dt.hour
    merged["day"] = merged["timestamp_utc"].dt.day
    merged["month"] = merged["timestamp_utc"].dt.month
    merged["dow"] = merged["timestamp_utc"].dt.weekday
    merged["is_weekend"] = merged["dow"].isin([5, 6]).astype(int)

    # 🕒 Use local Pakistan time for fetched_at
    pk_tz = pytz.timezone("Asia/Karachi")
    merged["fetched_at"] = datetime.datetime.now(pk_tz).strftime("%Y-%m-%d %H:%M:%S")

    merged["aqi_change_rate"] = 0
    merged["o3"] = merged["ozone"]
    merged["no2"] = merged["nitrogen_dioxide"]
    merged["so2"] = merged["sulphur_dioxide"]
    merged["co"] = merged["carbon_monoxide"]

    # 🧾 Reorder columns to exactly match your main dataset
    ordered_cols = [
        "timestamp_utc","temp_C","feels_like_C","humidity_%","pressure_hPa","wind_speed_m/s","clouds_%","weather",
        "pm10","pm25","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone",
        "temp","feels_like","humidity","pressure","wind_speed","clouds",
        "hour","day","month","dow","is_weekend","fetched_at","station_name","aqi_change_rate",
        "aqi_api","o3","no2","so2","co","wind_deg","weather_desc"
    ]

    merged = merged.reindex(columns=ordered_cols, fill_value=None)

    print(f"\n🧩 Merged DataFrame Preview ({len(merged.columns)} columns):\n")
    print(merged.head())

# ----------------------------------------------------------
# 🏁 Run
# ----------------------------------------------------------
if __name__ == "__main__":
    preview_merge()
