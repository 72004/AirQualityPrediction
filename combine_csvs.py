# combine_simple_asof.py
import pandas as pd

WEATHER_CSV = "karachi_weather_20days_hourly.csv"
AIR_CSV = "karachi_air_quality_20days_hourly.csv"
OUT_CSV = "testing.csv"

# Load
w = pd.read_csv(WEATHER_CSV)
a = pd.read_csv(AIR_CSV)

# Parse datetimes and align to hour
w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce").dt.floor("H")
a["datetime"] = pd.to_datetime(a["datetime"], errors="coerce").dt.floor("H")

# Drop rows with bad datetimes
w = w.dropna(subset=["datetime"]).sort_values("datetime")
a = a.dropna(subset=["datetime"]).sort_values("datetime")

# Remove duplicate datetimes, keeping the last (adjust as needed)
w = w.drop_duplicates(subset=["datetime"], keep="last")
a = a.drop_duplicates(subset=["datetime"], keep="last")

# Merge on nearest hour within 1h tolerance
combined = pd.merge_asof(
    w, a,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1h"),
    suffixes=("_weather", "_air")
)

# Optional: show diagnostics
print(f"Weather rows: {len(w)}, Air rows: {len(a)}, Combined rows: {len(combined)}")
print("Unmatched (no air match):", combined["pm10"].isna().sum() if "pm10" in combined else "n/a")

# Save
combined.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV}")