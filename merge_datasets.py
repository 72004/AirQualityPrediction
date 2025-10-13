import pandas as pd
import datetime
import hopsworks
from hsfs.feature import Feature

# ----------------------------------------------------------
# Function to store merged dataset into Hopsworks Feature Store
# ----------------------------------------------------------
def store_merged_to_hopsworks(df, city):
    import hopsworks
    from hsfs.feature import Feature

    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 🧹 Step 1: Clean column names (to fix the humidity_% issue)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace('%', 'percent', regex=False)
                  .str.replace(r'[^a-z0-9_]', '_', regex=True)
    )

    print("🧾 Cleaned column names:", df.columns.tolist())

    # ✅ Step 2: Define schema (you already have it correctly)
    features = [
        Feature("timestamp_utc", "timestamp"),
        Feature("temp", "double"),
        Feature("feels_like", "double"),
        Feature("humidity", "int"),
        Feature("pressure", "int"),
        Feature("wind_speed", "double"),
        Feature("clouds", "int"),
        Feature("weather", "string"),
        Feature("pm10", "double"),
        Feature("pm25", "double"),
        Feature("carbon_monoxide", "double"),
        Feature("nitrogen_dioxide", "double"),
        Feature("sulphur_dioxide", "double"),
        Feature("ozone", "double"),
        Feature("hour", "int"),
        Feature("day", "int"),
        Feature("month", "int"),
        Feature("dow", "int"),
        Feature("is_weekend", "int"),
        Feature("fetched_at", "string"),
        Feature("station_name", "string"),
        Feature("aqi_change_rate", "double"),
        Feature("aqi_api", "int"),
        Feature("o3", "int"),
        Feature("no2", "int"),
        Feature("so2", "int"),
        Feature("co", "int"),
        Feature("wind_deg", "int"),
        Feature("weather_desc", "string"),
    ]

    df["record_id"] = df.index.astype(str) + "_" + city.lower()

    # ✅ Step 3: Create or get Feature Group
    fg = fs.get_or_create_feature_group(
        name=f"{city.lower()}_20days_weather_air_quality_merged",
        version=1,
        description=f"Merged weather and air quality data from OpenWeather and OpenMeteo APIs for {city}",
        primary_key=["record_id"],  # 👈 use synthetic key
        event_time="timestamp_utc",
        online_enabled=True  # still enabled for real-time queries
    )

    print("📤 Inserting merged dataset into Hopsworks Feature Group...")
    fg.insert(df, write_options={"wait_for_job": False})
    print("✅ Successfully inserted merged dataset into Hopsworks!")

# ----------------------------------------------------------
# Step 1: Load Datasets
# ----------------------------------------------------------
city = "Karachi"
df_weather = pd.read_csv("karachi_weather_20days_hourly.csv")
df_aq = pd.read_csv("karachi_air_quality_20days_hourly.csv")

# ----------------------------------------------------------
# Step 2: Convert datetime and sort
# ----------------------------------------------------------
df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
df_aq["datetime"] = pd.to_datetime(df_aq["datetime"])
df_weather.sort_values("datetime", inplace=True)
df_aq.sort_values("datetime", inplace=True)

# ----------------------------------------------------------
# Step 3: Merge both datasets by nearest hour
# ----------------------------------------------------------
merged = pd.merge_asof(
    df_weather,
    df_aq,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)

# Drop nulls in pollutants (optional)
merged.dropna(subset=["pm2_5", "pm10"], inplace=True)
merged.sort_values("datetime", ascending=False, inplace=True)

# ----------------------------------------------------------
# Step 4: Rename columns
# ----------------------------------------------------------
merged.rename(columns={
    "datetime": "timestamp_utc",
    "pm2_5": "pm25",
    "description": "weather_desc"
}, inplace=True)

# ----------------------------------------------------------
# Step 5: Add missing columns (in case not present)
# ----------------------------------------------------------
expected_columns = [
    "timestamp_utc", "temp", "feels_like", "humidity", "pressure",
    "wind_speed", "clouds", "weather", "pm10", "pm25",
    "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "hour", "day", "month", "dow", "is_weekend", "fetched_at",
    "station_name", "aqi_change_rate", "aqi_api", "o3", "no2",
    "so2", "co", "wind_deg", "weather_desc"
]

for col in expected_columns:
    if col not in merged.columns:
        merged[col] = 0 if col not in ["weather", "station_name", "weather_desc", "fetched_at"] else None

# ----------------------------------------------------------
# Step 6: Time-based features
# ----------------------------------------------------------
merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"])
merged["hour"] = merged["timestamp_utc"].dt.hour
merged["day"] = merged["timestamp_utc"].dt.day
merged["month"] = merged["timestamp_utc"].dt.month
merged["dow"] = merged["timestamp_utc"].dt.weekday
merged["is_weekend"] = merged["dow"].isin([5, 6]).astype(int)
merged["fetched_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
merged["station_name"] = city

# ----------------------------------------------------------
# Step 7: Convert numeric columns safely
# ----------------------------------------------------------
numeric_cols = [
    "temp", "feels_like", "humidity", "pressure", "wind_speed", "clouds",
    "pm10", "pm25", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "hour", "day", "month", "dow", "is_weekend", "aqi_change_rate",
    "aqi_api", "o3", "no2", "so2", "co", "wind_deg"
]

for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

# ----------------------------------------------------------
# Step 8: Save Locally
# ----------------------------------------------------------
merged.to_csv("karachi_20days_weather_air_quality_merged.csv", index=False)
print(f"✅ Merged dataset saved → 'karachi_weather_air_quality_merged.csv'")
print(f"📊 Total merged records: {len(merged)}")
print(f"🕒 Range: {merged['timestamp_utc'].min()} → {merged['timestamp_utc'].max()}")

# ----------------------------------------------------------
# Step 9: Push to Hopsworks
# ----------------------------------------------------------
# 🧹 Drop problematic column
merged = merged.drop(columns=['weather_desc'], errors='ignore')

# 🚀 Insert merged dataset into Hopsworks
store_merged_to_hopsworks(merged, city)


