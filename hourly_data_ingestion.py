import requests
import pandas as pd
import datetime
import hopsworks
from hsfs.feature import Feature
import os
from dotenv import load_dotenv
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
        # remove microseconds at source
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
        "pm25": latest["pm2_5"],
        "carbon_monoxide": latest["carbon_monoxide"],
        "nitrogen_dioxide": latest["nitrogen_dioxide"],
        "sulphur_dioxide": latest["sulphur_dioxide"],
        "ozone": latest["ozone"],
        "aqi_api": latest["us_aqi"],
        "station_name": "Karachi"
    }

    return pd.DataFrame([aqi_data])

# ----------------------------------------------------------
# 🧰 Align schema to expected Feature Store columns and dtypes
# ----------------------------------------------------------
def align_schema(df: pd.DataFrame, city: str) -> pd.DataFrame:
    df = df.copy()

    # Ensure station_name present and string (avoid NaN floats in Avro)
    if "station_name" not in df:
        df["station_name"] = city
    df["station_name"] = df["station_name"].fillna(city).astype(str)

    # Ensure fetched_at is string
    if "fetched_at" in df:
        df["fetched_at"] = df["fetched_at"].astype(str)

    # Canonical weather columns expected by FS
    if "temp_c" not in df:
        if "temp" in df:
            df["temp_c"] = pd.to_numeric(df["temp"], errors="coerce")
        else:
            df["temp_c"] = pd.NA
    if "feels_like_c" not in df:
        if "feels_like" in df:
            df["feels_like_c"] = pd.to_numeric(df["feels_like"], errors="coerce")
        else:
            df["feels_like_c"] = pd.NA
    if "humidity_percent" not in df:
        if "humidity" in df:
            df["humidity_percent"] = pd.to_numeric(df["humidity"], errors="coerce").astype("Int64")
        else:
            df["humidity_percent"] = pd.NA
    if "pressure_hpa" not in df:
        if "pressure" in df:
            df["pressure_hpa"] = pd.to_numeric(df["pressure"], errors="coerce").astype("Int64")
        else:
            df["pressure_hpa"] = pd.NA
    if "wind_speed_m_s" not in df:
        if "wind_speed" in df:
            df["wind_speed_m_s"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        else:
            df["wind_speed_m_s"] = pd.NA
    if "clouds_percent" not in df:
        if "clouds" in df:
            df["clouds_percent"] = pd.to_numeric(df["clouds"], errors="coerce").astype("Int64")
        else:
            df["clouds_percent"] = pd.NA

    # Pollutants: map to expected short names and cast to integers if required
    if "ozone" in df and "o3" not in df:
        df["o3"] = pd.to_numeric(df["ozone"], errors="coerce").round().astype("Int64")
    if "nitrogen_dioxide" in df and "no2" not in df:
        df["no2"] = pd.to_numeric(df["nitrogen_dioxide"], errors="coerce").round().astype("Int64")
    if "sulphur_dioxide" in df and "so2" not in df:
        df["so2"] = pd.to_numeric(df["sulphur_dioxide"], errors="coerce").round().astype("Int64")
    if "carbon_monoxide" in df and "co" not in df:
        df["co"] = pd.to_numeric(df["carbon_monoxide"], errors="coerce").round().astype("Int64")

    # Ensure aqi_api is integer if FS expects bigint
    if "aqi_api" in df:
        df["aqi_api"] = pd.to_numeric(df["aqi_api"], errors="coerce").round().astype("Int64")

    # If FS expects raw temp/feels_like/wind_speed as bigint, cast them
    for col in ["temp", "feels_like", "wind_speed"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    return df

# ----------------------------------------------------------
# 🧠 Store merged data into Hopsworks Feature Store
# ----------------------------------------------------------
def store_to_hopsworks(df, city):
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Ensure station_name available before alignment
    if "station_name" not in df:
        df["station_name"] = city

    # Align schema and types before any normalization
    df = align_schema(df, city)

    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace('%', 'percent', regex=False)
                  .str.replace(r'[^a-z0-9_]', '_', regex=True)
    )

    fg = fs.get_or_create_feature_group(
        name=f"{city.lower()}_20days_weather_air_quality_merged",
        version=1,
        description=f"Live weather and air quality data for {city}",
        primary_key=["record_id"],
        event_time="timestamp_utc",
        online_enabled=True
    )

    print("📤 Inserting latest record into Hopsworks...")
    fg.insert(df, write_options={"wait_for_job": False})
    print("✅ Successfully inserted latest record!")

# ----------------------------------------------------------
# ⚙️ Main hourly pipeline
# ----------------------------------------------------------
def run_hourly_pipeline():
    CITY = "Karachi"
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

    # 1️⃣ Fetch new weather + AQI data
    df_weather = fetch_weather(OPENWEATHER_API_KEY, CITY)
    df_aqi = fetch_aqi()

    # 2️⃣ Merge both by nearest timestamp
    merged = pd.merge_asof(
        df_weather.sort_values("timestamp_utc"),
        df_aqi.sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta("1h")
    )

    # 3️⃣ Add engineered features
    merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"]).dt.floor("S")
    merged["hour"] = merged["timestamp_utc"].dt.hour
    merged["day"] = merged["timestamp_utc"].dt.day
    merged["month"] = merged["timestamp_utc"].dt.month
    merged["dow"] = merged["timestamp_utc"].dt.weekday
    merged["is_weekend"] = merged["dow"].isin([5, 6]).astype(int)
    merged["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged["aqi_change_rate"] = 0  # placeholder for future

    # Ensure station_name is set deterministically as in merge_datasets.py
    merged["station_name"] = CITY

    merged["record_id"] = merged.index.astype(str) + "_" + CITY.lower()

    # 4️⃣ Write CSV matching merge_datasets.py expected header and order, prepend new rows
    csv_columns_order = [
        "timestamp_utc", "temp", "feels_like", "humidity", "pressure",
        "wind_speed", "clouds", "weather", "pm10", "pm25",
        "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
        "hour", "day", "month", "dow", "is_weekend", "fetched_at",
        "station_name", "aqi_change_rate", "aqi_api", "o3", "no2", "so2", "co", "wind_deg", "weather_desc"
    ]

    csv_df = merged.copy()
    # Ensure all required columns exist for ordering
    for col in csv_columns_order:
        if col not in csv_df:
            csv_df[col] = pd.NA if col in ["weather", "weather_desc", "station_name", "fetched_at"] else 0

    # Order and format
    csv_df = csv_df[csv_columns_order]
    csv_df["timestamp_utc"] = pd.to_datetime(csv_df["timestamp_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Coerce numeric types and cast int-like for consistency
    non_numeric = {"weather", "weather_desc", "station_name", "fetched_at", "timestamp_utc"}
    numeric_cols = [c for c in csv_df.columns if c not in non_numeric]
    csv_df[numeric_cols] = csv_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    int_like = ["hour", "day", "month", "dow", "is_weekend", "aqi_api", "o3", "no2", "so2", "co", "wind_deg", "clouds", "humidity", "pressure"]
    for col in int_like:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].astype(int)
    csv_df["station_name"] = csv_df["station_name"].fillna(CITY).astype(str)
    csv_df["weather_desc"] = csv_df["weather_desc"].fillna("").astype(str)

    merged_csv_path = "karachi_20days_weather_air_quality_merged.csv"
    if os.path.exists(merged_csv_path):
        existing_df = pd.read_csv(merged_csv_path)
        # Align existing columns to desired order, add missing, drop extras
        for col in csv_columns_order:
            if col not in existing_df.columns:
                existing_df[col] = pd.NA if col in ["weather", "weather_desc", "station_name", "fetched_at"] else 0
        existing_df = existing_df[csv_columns_order]
        # Normalize existing rows formatting/types
        existing_df["timestamp_utc"] = pd.to_datetime(existing_df["timestamp_utc"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        ex_numeric_cols = [c for c in existing_df.columns if c not in non_numeric]
        existing_df[ex_numeric_cols] = existing_df[ex_numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        for col in int_like:
            if col in existing_df.columns:
                existing_df[col] = existing_df[col].astype(int)
        existing_df["station_name"] = existing_df["station_name"].fillna(CITY).astype(str)
        if "weather_desc" in existing_df.columns:
            existing_df["weather_desc"] = existing_df["weather_desc"].fillna("").astype(str)
        # Prepend new rows
        out_df = pd.concat([csv_df, existing_df], ignore_index=True)
    else:
        out_df = csv_df

    # Always write with header, overwrite file
    out_df.to_csv(merged_csv_path, mode="w", header=True, index=False)
    print(f"📁 Wrote (prepended) data to {merged_csv_path}")

    # 5️⃣ Push to Hopsworks (drop weather_desc like merge_datasets.py)
    merged_for_store = merged.drop(columns=["weather_desc"], errors="ignore")
    store_to_hopsworks(merged_for_store, CITY)

    print("🌍 Hourly ingestion complete at:", datetime.datetime.now())

# ----------------------------------------------------------
# 🕐 Run the script manually or via cron
# ----------------------------------------------------------
# if __name__ == "__main__":
run_hourly_pipeline()
