import os
import pandas as pd
import datetime
import requests
import hopsworks
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------------------------------
# 🌤 Fetch latest weather data from OpenWeather API
# ----------------------------------------------------------
def fetch_weather(api_key, city="Karachi"):
    print("\n🔹 Fetching latest weather data...")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    data = response.json()

    weather_data = {
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "clouds": data["clouds"]["all"],
        "weather": data["weather"][0]["main"]
    }

    print("✅ Weather data fetched successfully:")
    for k, v in weather_data.items():
        print(f"   {k}: {v}")

    return pd.DataFrame([weather_data])

# ----------------------------------------------------------
# 🌫 Fetch latest air quality data from Open-Meteo API
# ----------------------------------------------------------
def fetch_aqi(lat=24.8607, lon=67.0011):
    print("\n🔹 Fetching latest air quality data...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide",
                   "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
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
        "ozone": latest["ozone"]
    }

    print("✅ AQI data fetched successfully:")
    for k, v in aqi_data.items():
        print(f"   {k}: {v}")

    return pd.DataFrame([aqi_data])

# ----------------------------------------------------------
# 🚀 Append new record to CSV and Hopsworks
# ----------------------------------------------------------
def append_to_csv_and_hopsworks(df_combined, output_file):
    print("\n-----------------------------")
    print("📂 Appending data to CSV & Hopsworks...")
    print("-----------------------------")

    # ✅ Append to CSV
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df_final = pd.concat([existing_df, df_combined], ignore_index=True)
    else:
        df_final = df_combined

    df_final.to_csv(output_file, index=False)
    print(f"✅ Data appended locally → {output_file}")
    print("🧾 Last row appended:")
    print(df_combined.to_string(index=False))

    # ✅ Prepare datatypes for Hopsworks
    df_combined["datetime"] = pd.to_datetime(df_combined["datetime"])
    numeric_cols = [
        "temp_c", "feels_like_c", "humidity_pct", "pressure_hpa",
        "wind_speed_m_per_s", "clouds_pct",
        "pm10", "pm2_5", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide", "ozone"
    ]
    for col in numeric_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

    # ✅ Insert into Hopsworks
    print("\n🚀 Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    try:
        feature_group = fs.get_feature_group(name="testing_2", version=1)
        print("📦 Found existing feature group 'testing_2'.")
    except:
        print("🆕 Creating new feature group 'testing_2'...")
        feature_group = fs.create_feature_group(
            name="testing_2",
            version=1,
            description="Hourly weather + AQI data for Karachi",
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=False
        )

    print("\n🧾 Schema preview before insert:")
    print(df_combined.dtypes)

    feature_group.insert(df_combined)
    print("✅ Successfully inserted new record into Hopsworks feature group 'testing_2'.")


# ----------------------------------------------------------
# ⚙ Main hourly pipeline
# ----------------------------------------------------------
def run_hourly_pipeline():
    print("\n==============================")
    print("🚀 Starting Hourly Data Pipeline")
    print("==============================")

    CITY = "Karachi"
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FILE = os.path.join(BASE_DIR, "testing_2.csv")

    df_weather = fetch_weather(OPENWEATHER_API_KEY, CITY)
    df_aqi = fetch_aqi()

    w = df_weather.iloc[0]
    a = df_aqi.iloc[0]
    combined_data = {
        "datetime": pd.to_datetime(w["timestamp_utc"]),
        "temp_c": float(w["temp"]),
        "feels_like_c": float(w["feels_like"]),
        "humidity_pct": float(w["humidity"]),
        "pressure_hpa": float(w["pressure"]),
        "wind_speed_m_per_s": float(w["wind_speed"]),
        "clouds_pct": float(w["clouds"]),
        "weather": str(w["weather"]),
        "pm10": float(a["pm10"]),
        "pm2_5": float(a["pm2_5"]),
        "carbon_monoxide": float(a["carbon_monoxide"]),
        "nitrogen_dioxide": float(a["nitrogen_dioxide"]),
        "sulphur_dioxide": float(a["sulphur_dioxide"]),
        "ozone": float(a["ozone"])
    }

    df_combined = pd.DataFrame([combined_data])
    print("\n🌦 Final Combined Weather + AQI Record:")
    print(df_combined.to_string(index=False))

    append_to_csv_and_hopsworks(df_combined, OUTPUT_FILE)
# ----------------------------------------------------------
# 🏁 Run the pipeline
# ----------------------------------------------------------
if __name__ == "__main__":
    run_hourly_pipeline()
