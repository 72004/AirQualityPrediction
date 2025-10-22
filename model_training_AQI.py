
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks
from dotenv import load_dotenv
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# ============================================
# 1️⃣ Load Environment Variables
# ============================================
load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY not found. Please add it to your .env file.")

# ============================================
# 2️⃣ Connect to Hopsworks
# ============================================
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

# ============================================
# 3️⃣ Fetch Data from Feature Store
# ============================================
print("📦 Fetching feature group data...")
feature_group = fs.get_feature_group(name="weather_data_2", version=1)
df = feature_group.read()
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# 4️⃣ AQI Calculation
# ============================================
def calculate_aqi(pm25, pm10):
    """Simple AQI formula for demonstration."""
    return 0.5 * pm25 + 0.5 * pm10

df['aqi'] = calculate_aqi(df['pm2_5'], df['pm10'])
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').dropna()

# ============================================
# 5️⃣ Feature Engineering
# ============================================
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month

def create_lagged_features(data, lag=24):
    """Create lag features for past `lag` hours."""
    for i in range(1, lag + 1):
        data[f"aqi_lag_{i}"] = data['aqi'].shift(i)
    return data.dropna()

df = create_lagged_features(df)

# ============================================
# 6️⃣ Prepare Features and Target
# ============================================
features = [
    'temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'ozone',
    'hour', 'day', 'month'
] + [f"aqi_lag_{i}" for i in range(1, 25)]

X = df[features]
y = df['aqi']

# ============================================
# 7️⃣ Train/Test Split
# ============================================
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ============================================
# 8️⃣ Train Random Forest Model
# ============================================
print("🧠 Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# ============================================
# 9️⃣ Evaluation Metrics
# ============================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

def regression_accuracy(y_true, y_pred, tolerance=0.1):
    """Custom metric: % predictions within ±10% of true value."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    within_range = np.abs((y_true - y_pred) / y_true) <= tolerance
    return np.mean(within_range) * 100

accuracy = regression_accuracy(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")
print(f"Accuracy (±10%): {accuracy:.2f}%")

# ============================================
# 🔟 Save Model Locally
# ============================================
model_dir = "aqi_rf_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "rf_aqi_forecast.pkl")
joblib.dump(rf_model, model_path)
print(f"\n💾 Model saved locally at: {model_path}")

# ============================================
# 11️⃣ Register Model in Hopsworks
# ============================================
print("🚀 Registering model in Hopsworks...")
mr = project.get_model_registry()

model_meta = mr.python.create_model(
    name="AQI_RF_Forecaster",
    metrics={"MAE": mae, "RMSE": rmse, "R2": r2, "Accuracy": accuracy},
    description="Random Forest model for 3-day AQI forecasting",
)
model_meta.save(model_dir)
print("✅ Model saved successfully in Hopsworks Model Registry!")

# ============================================
# 12️⃣ Forecast Next 3 Days (72 hours)
# ============================================
print("\n🔮 Generating 3-day AQI forecast...")
recent_data = df.tail(24).copy()
future_preds = []

for i in range(72):  # Predict next 72 hours
    row = recent_data.iloc[-1:].copy()

    # Build lag features
    for j in range(1, 25):
        row[f"aqi_lag_{j}"] = recent_data['aqi'].iloc[-j]

    X_future = row[features]
    pred = rf_model.predict(X_future)[0]
    future_preds.append(pred)

    new_row = {
        'datetime': row['datetime'].iloc[0] + pd.Timedelta(hours=1),
        'aqi': pred,
        'temperature_2m': row['temperature_2m'].iloc[0],
        'relative_humidity_2m': row['relative_humidity_2m'].iloc[0],
        'windspeed_10m': row['windspeed_10m'].iloc[0],
        'ozone': row['ozone'].iloc[0],
        'hour': (row['hour'].iloc[0] + 1) % 24,
        'day': row['day'].iloc[0],
        'month': row['month'].iloc[0],
    }
    recent_data = pd.concat([recent_data, pd.DataFrame([new_row])])

print("✅ AQI forecast generated for next 3 days.")
print(f"Sample predictions: {future_preds[:10]} ...")

# ============================================
# 13️⃣ Visualization (Optional)
# ============================================
try:
    print("📈 Generating Plotly visualization...")
    df_recent = df.tail(72*3).copy()  # past 3 days
    actual_dates = df_recent['datetime']
    actual_aqi = df_recent['aqi']

    last_timestamp = df_recent['datetime'].iloc[-1]
    future_dates = [last_timestamp + timedelta(hours=i+1) for i in range(72)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_aqi,
        mode='lines+markers',
        name='Actual AQI (Past 3 Days)',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        mode='lines+markers',
        name='Predicted AQI (Next 3 Days)',
        line=dict(color='orange', width=2, dash='dot')
    ))
    fig.update_layout(
        title='🌤 AQI Forecast: Past vs Next 3 Days (Random Forest)',
        xaxis_title='Date & Time',
        yaxis_title='Air Quality Index (AQI)',
        legend=dict(x=0, y=1.1, orientation="h"),
        template='plotly_white',
        hovermode='x unified'
    )
    fig.show()
except Exception as e:
    print(f"⚠️ Visualization skipped: {e}")
