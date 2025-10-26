# ==========================================
# 🌍 AQI Random Forest Model Training + Push to Hopsworks
# ==========================================

import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from math import sqrt
import os
from dotenv import load_dotenv

# ==========================================
# 🔐 LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY not found in .env file!")

# ==========================================
# 🔑 CONNECT TO HOPSWORKS
# ==========================================
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ==========================================
# 📥 LOAD FEATURE GROUP
# ==========================================
fg = fs.get_feature_group(name="weather_data_2", version=1)
df = fg.read()
print("✅ Data loaded from Hopsworks:", df.shape)

# ==========================================
# 🧮 AQI CALCULATION (EPA FORMULA)
# ==========================================
def calculate_aqi_subindex(C, breakpoints):
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= C <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (C - Clow) + Ilow
    return None

def calculate_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    co_ppm = co / 1145 if co is not None else None
    no2_ppb = no2 / 1.88 if no2 is not None else None
    so2_ppb = so2 / 2.62 if so2 is not None else None
    o3_ppb = o3 / 2.0 if o3 is not None else None

    pm25_bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]

    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]

    o3_bp = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300)
    ]

    no2_bp = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300)
    ]

    so2_bp = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300)
    ]

    co_bp = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300)
    ]

    subindices = {}
    if pm25 is not None: subindices["PM2.5"] = calculate_aqi_subindex(pm25, pm25_bp)
    if pm10 is not None: subindices["PM10"] = calculate_aqi_subindex(pm10, pm10_bp)
    if o3_ppb is not None: subindices["O3"] = calculate_aqi_subindex(o3_ppb, o3_bp)
    if no2_ppb is not None: subindices["NO2"] = calculate_aqi_subindex(no2_ppb, no2_bp)
    if so2_ppb is not None: subindices["SO2"] = calculate_aqi_subindex(so2_ppb, so2_bp)
    if co_ppm is not None: subindices["CO"] = calculate_aqi_subindex(co_ppm, co_bp)

    subindices = {k: v for k, v in subindices.items() if v is not None}
    if not subindices:
        return np.nan, "N/A"

    dominant_pollutant = max(subindices, key=subindices.get)
    aqi_value = round(subindices[dominant_pollutant], 1)
    return aqi_value, dominant_pollutant

# Apply AQI calculation
df["aqi"], df["dominant_pollutant"] = zip(*df.apply(
    lambda row: calculate_aqi(row["pm2_5"], row["pm10"],
                              row["ozone"], row["nitrogen_dioxide"],
                              row["sulphur_dioxide"], row["carbon_monoxide"]),
    axis=1
))
print("✅ AQI calculated successfully")

# ==========================================
# 🧠 FEATURE ENGINEERING
# ==========================================
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").dropna()

# Time-based features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

# Lag features
def create_lagged_features(data, lag=24):
    data = data.copy()
    for i in range(1, lag + 1):
        data[f"aqi_lag_{i}"] = data["aqi"].shift(i)
    return data.dropna()

df = create_lagged_features(df, lag=24)

# ==========================================
# 🎯 FEATURES AND TARGET
# ==========================================
features = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "surface_pressure", "cloud_cover",
    "windspeed_10m", "winddirection_10m", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index",
    "hour", "day", "month"
] + [f"aqi_lag_{i}" for i in range(1, 25)]

X = df[features]
y = df["aqi"]

# ==========================================
# ✂️ TRAIN / TEST SPLIT
# ==========================================
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==========================================
# 🌲 TRAIN MODEL
# ==========================================
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# ==========================================
# 📈 EVALUATE MODEL
# ==========================================
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f} | R² (Accuracy): {r2*100:.2f}%")

# ==========================================
# 💾 SAVE AND PUSH MODEL TO HOPSWORKS
# ==========================================
mr = project.get_model_registry()

model_dir = "aqi_model_push"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "rf_aqi_forecast_V2.pkl")
joblib.dump(rf_model, model_path)

metrics = {
    "MAE": round(mae, 3),
    "RMSE": round(rmse, 3),
    "R2": round(r2, 3),
    "Accuracy (%)": round(r2 * 100, 2)
}

model_name = "AQI_RF_Forecaster_V2"
model_desc = f"Random Forest model for AQI forecasting — trained {datetime.now().strftime('%Y-%m-%d %H:%M')}"

model_meta = mr.python.create_model(
    name=model_name,
    metrics=metrics,
    description=model_desc,
    input_example=X_test.iloc[:1],
    model_schema=None
)

model_meta.save(model_dir)
print(f"✅ Model '{model_name}' registered successfully in Hopsworks!")
print(f"📈 Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2*100:.2f}%")
