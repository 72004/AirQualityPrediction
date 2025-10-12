import pandas as pd

# Load both datasets
df_weather = pd.read_csv("karachi_weather_20days_hourly.csv")
df_aq = pd.read_csv("karachi_air_quality_20days_hourly.csv")

# Convert to datetime
df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
df_aq["datetime"] = pd.to_datetime(df_aq["datetime"])

# Sort chronologically (oldest → newest)
df_weather.sort_values("datetime", inplace=True)
df_aq.sort_values("datetime", inplace=True)

# Merge by nearest timestamp (tolerance = 1 hour)
merged = pd.merge_asof(
    df_weather,
    df_aq,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)

# Drop rows with missing pollutant data (optional)
merged.dropna(subset=["pm2_5", "pm10"], inplace=True)

# Sort descending by datetime (newest → oldest)
merged.sort_values("datetime", ascending=False, inplace=True)

# Save final merged file
merged.to_csv("karachi_weather_air_quality_merged.csv", index=False)

print(f"✅ Merged dataset saved → 'karachi_weather_air_quality_merged.csv'")
print(f"📊 Total merged records: {len(merged)}")
print("\n🕒 Sample timestamps range:")
print(merged['datetime'].min(), "→", merged['datetime'].max())
