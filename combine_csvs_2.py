# combine_simple_asof.py
import pandas as pd

# ------------------------------
# 📂 Input and Output Paths
# ------------------------------
WEATHER_CSV = "karachi_weather_20days_hourly.csv"
AIR_CSV = "karachi_air_quality_20days_hourly.csv"
OUT_CSV = "testing_2.csv"

# ------------------------------
# 🧭 Load and Preprocess
# ------------------------------
w = pd.read_csv(WEATHER_CSV)
a = pd.read_csv(AIR_CSV)

# Parse datetimes and floor to hour
w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce").dt.floor("H")
a["datetime"] = pd.to_datetime(a["datetime"], errors="coerce").dt.floor("H")

# Drop rows with invalid datetime
w = w.dropna(subset=["datetime"]).sort_values("datetime")
a = a.dropna(subset=["datetime"]).sort_values("datetime")

# Remove duplicate datetimes (keep latest)
w = w.drop_duplicates(subset=["datetime"], keep="last")
a = a.drop_duplicates(subset=["datetime"], keep="last")

# ------------------------------
# 🔗 Merge on nearest hour (within 1h tolerance)
# ------------------------------
combined = pd.merge_asof(
    w, a,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1h"),
    suffixes=("_weather", "_air")
)

# ------------------------------
# 🧹 Standardize Final Column Names (match existing CSV)
# ------------------------------
rename_map = {
    # Weather columns
    "temp": "temp_C",
    "feels_like": "feels_like_C",
    "humidity": "humidity_%",
    "pressure": "pressure_hPa",
    "wind_speed": "wind_speed_m/s",
    "clouds": "clouds_%",
    "weather": "weather",
    # Air quality columns
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "carbon_monoxide": "carbon_monoxide",
    "nitrogen_dioxide": "nitrogen_dioxide",
    "sulphur_dioxide": "sulphur_dioxide",
    "ozone": "ozone"
}

combined = combined.rename(columns=rename_map)

# Keep only the final standardized columns (in correct order)
final_columns = [
    "datetime",
    "temp_C",
    "feels_like_C",
    "humidity_%",
    "pressure_hPa",
    "wind_speed_m/s",
    "clouds_%",
    "weather",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
]

# Filter only columns that exist in DataFrame
combined = combined[[col for col in final_columns if col in combined.columns]]

# ------------------------------
# 🧾 Diagnostics
# ------------------------------
print(f"✅ Weather rows: {len(w)}, Air rows: {len(a)}, Combined rows: {len(combined)}")
if "pm10" in combined:
    print(f"Unmatched (no air match): {combined['pm10'].isna().sum()}")
    
def sanitize_column(col):
    col = col.lower()
    col = col.replace("%", "_pct")
    col = col.replace("/", "_per_")
    col = col.replace(" ", "_")
    col = col.replace("__", "_")  # clean double underscores
    return col

combined.columns = [sanitize_column(c) for c in combined.columns]

# ------------------------------
# 💾 Save Clean Output
# ------------------------------
combined.to_csv(OUT_CSV, index=False)
print(f"✅ Saved clean merged file with correct headers: {OUT_CSV}")
