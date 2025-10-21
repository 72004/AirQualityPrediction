import hopsworks
import os
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------
# 1️⃣ Connect to Hopsworks and Fetch Latest Data
# ----------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------------------------------
# 1️⃣ Connect to Hopsworks
# ----------------------------------------------------------
print("🔗 Connecting to Hopsworks...")

# Try to get key from environment
api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key or api_key.strip() == "":
    print("⚠️ HOPSWORKS_API_KEY not found in environment. Trying login prompt...")
    project = hopsworks.login()  # fallback for local testing
else:
    print("✅ Using API key from environment")
    project = hopsworks.login(api_key_value=api_key)

fs = project.get_feature_store()
fg = fs.get_feature_group("testing_2", version=1)
df = fg.read()

print(f"✅ Loaded {len(df)} records from Feature Store.")
print(df.head())

# ----------------------------------------------------------
# 2️⃣ Data Preprocessing (same as training)
# ----------------------------------------------------------
df = df.drop(columns=["datetime", "weather"], errors="ignore")
df = df.fillna(df.mean())

X = df.drop(columns=["pm2_5"])
y = df["pm2_5"]

# ----------------------------------------------------------
# 3️⃣ Load Best Model from Hopsworks Model Registry
# ----------------------------------------------------------
print("📦 Loading model from Hopsworks Model Registry...")
mr = project.get_model_registry()
model_meta = mr.get_model("aqi_forecast_model", version=1)
model_dir = model_meta.download()

try:
    model = joblib.load(model_dir + "/best_aqi_model.pkl")
except:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_dir + "/best_aqi_model_nn.h5")

print("✅ Model loaded successfully.")

# ----------------------------------------------------------
# 4️⃣ Evaluate on Historical (Unseen) Data
# ----------------------------------------------------------
y_pred = model.predict(X)
if y_pred.ndim > 1:
    y_pred = y_pred.flatten()

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
accuracy = (1 - (mae / y.mean())) * 100

print("\n📊 Model Evaluation Metrics:")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
print(f"🎯 Accuracy: {accuracy:.2f}%")

# ----------------------------------------------------------
# 5️⃣ Fetch 72-Hour Weather Forecast (OpenWeather)
# ----------------------------------------------------------
print("\n🌦 Fetching next 72-hour weather forecast...")

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 24.8607, 67.0011  # Karachi

url = f"https://api.openweathermap.org/data/2.5/forecast"
params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"}
response = requests.get(url, params=params)
data = response.json()

forecast_list = []
for item in data["list"]:
    timestamp = datetime.datetime.utcfromtimestamp(item["dt"])
    temp = item["main"]["temp"]
    feels_like = item["main"]["feels_like"]
    humidity = item["main"]["humidity"]
    pressure = item["main"]["pressure"]
    wind_speed = item["wind"]["speed"]
    clouds = item["clouds"]["all"]
    weather_desc = item["weather"][0]["description"]

    forecast_list.append({
        "datetime": timestamp,
        "temp_c": temp,
        "feels_like_c": feels_like,
        "humidity_pct": humidity,
        "pressure_hpa": pressure,
        "wind_speed_m_per_s": wind_speed,
        "clouds_pct": clouds,
        "weather": weather_desc,
        # pollutant placeholders (to predict)
        "carbon_monoxide": 702.0,
        "nitrogen_dioxide": 41.0,
        "sulphur_dioxide": 14.6,
        "ozone": 42.0,
        "pm10": 43.8
    })

forecast_df = pd.DataFrame(forecast_list)
forecast_df["hour"] = forecast_df["datetime"].dt.hour
forecast_df["day"] = forecast_df["datetime"].dt.day
forecast_df["month"] = forecast_df["datetime"].dt.month

# Ensure column order matches model input
forecast_df = forecast_df.reindex(columns=X.columns, fill_value=0)

# ----------------------------------------------------------
# 6️⃣ Predict AQI for Next 72 Hours
# ----------------------------------------------------------
forecast_df = forecast_df.fillna(forecast_df.mean())
predicted_pm25 = model.predict(forecast_df)
if predicted_pm25.ndim > 1:
    predicted_pm25 = predicted_pm25.flatten()

# ----------------------------------------------------------
# 7️⃣ Visualize Forecast
# ----------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(forecast_list[0]["datetime"], predicted_pm25[0], 'bo', label="Predicted PM2.5")
plt.plot(range(len(predicted_pm25)), predicted_pm25, label="Predicted PM2.5 (72h)")
plt.title("🌫️ Predicted PM2.5 / AQI Trend for Next 72 Hours")
plt.xlabel("Forecast Hour")
plt.ylabel("Predicted PM2.5 Concentration")
plt.legend()
plt.grid(True)
plt.show()

print("\n✅ 72-hour AQI forecast completed successfully!")
