import requests
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

# ----------------------------------------------------------
# 🌤️ Fetch latest weather data from OpenWeather API
# ----------------------------------------------------------
def fetch_weather(api_key, city="Karachi"):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    weather_data = {
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "wind_deg": data["wind"].get("deg"),
        "clouds": data["clouds"]["all"],
        "weather": data["weather"][0]["main"],
        "weather_desc": data["weather"][0]["description"]
    }

    return pd.DataFrame([weather_data])


# ----------------------------------------------------------
# 🌫️ Fetch latest air quality data from Open-Meteo API
# ----------------------------------------------------------
def fetch_aqi(lat=24.8607, lon=67.0011):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ],
        "timezone": "UTC"
    }

    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    latest = df.iloc[-1]

    aqi_data = {
        "timestamp_utc": pd.to_datetime(latest["time"]),
        "pm10": latest["pm10"],
        "pm2_5": latest["pm2_5"],
        "carbon_monoxide": latest["carbon_monoxide"],
        "nitrogen_dioxide": latest["nitrogen_dioxide"],
        "sulphur_dioxide": latest["sulphur_dioxide"],
        "ozone": latest["ozone"],
        "aqi_api": latest["us_aqi"],
        "station_name": "karachi"
    }

    return pd.DataFrame([aqi_data])


# ----------------------------------------------------------
# ⚙️ Main hourly pipeline (Local + Hopsworks)
# ----------------------------------------------------------
def run_hourly_pipeline():
    CITY = "Karachi"
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FILE = os.path.join(BASE_DIR, "testing.csv")

    # Fetch both datasets
    df_weather = fetch_weather(OPENWEATHER_API_KEY, CITY)
    df_aqi = fetch_aqi()

    # Combine
    w = df_weather.iloc[0]
    a = df_aqi.iloc[0]

    combined_data = {
        "datetime": pd.to_datetime(w["timestamp_utc"]).strftime("%Y-%m-%d %H:%M:%S"),
        "temp_c": float(w.get("temp", 0) or 0),
        "feels_like_c": float(w.get("feels_like", 0) or 0),
        "humidity": int(w.get("humidity", 0) or 0),
        "pressure_hpa": int(w.get("pressure", 0) or 0),
        "wind_speed_ms": float(w.get("wind_speed", 0) or 0),
        "clouds": int(w.get("clouds", 0) or 0),
        "weather": str(w.get("weather", "") or ""),
        "pm10": float(a.get("pm10", 0) or 0),
        "pm2_5": float(a.get("pm2_5", 0) or 0),
        "carbon_monoxide": float(a.get("carbon_monoxide", 0) or 0),
        "nitrogen_dioxide": float(a.get("nitrogen_dioxide", 0) or 0),
        "sulphur_dioxide": float(a.get("sulphur_dioxide", 0) or 0),
        "ozone": float(a.get("ozone", 0) or 0)
    }

    df_combined = pd.DataFrame([combined_data])

    print("=== 🌦️ Final Combined Weather + AQI Data ===")
    print(df_combined.to_string(index=False))

    # ✅ Append to local CSV (at the END)
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        df_final = pd.concat([existing_df, df_combined], ignore_index=True)
    else:
        df_final = df_combined

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Data appended to local file: {OUTPUT_FILE}")

    # ------------------------------------------------------
    # ✅ Push to Hopsworks Feature Store
    # ------------------------------------------------------
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="karachi_weather_air_quality",
            version=1,
            primary_key=["datetime"],
            description="Hourly weather and air quality data for Karachi",
            online_enabled=True
        )

        fg.insert(df_combined)
        print("✅ Successfully inserted new record into Hopsworks feature group")

    except Exception as e:
        print(f"❌ Hopsworks upload failed: {e}")


# ----------------------------------------------------------
# 🕐 Run the script manually or via Hopsworks scheduler
# ----------------------------------------------------------
if __name__ == "__main__":
    run_hourly_pipeline()
