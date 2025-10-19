import hopsworks
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import joblib
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

# ----------------------------------------------------------
# 2️⃣ Load the Feature Group
# ----------------------------------------------------------
print("📥 Fetching data from Feature Store (testing_2)...")
feature_group = fs.get_feature_group(name="testing_2", version=1)
df = feature_group.read()

print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 3️⃣ Handle Missing Values
# ----------------------------------------------------------
pollutant_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
for col in pollutant_cols:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)

df.fillna(method="ffill", inplace=True)  # forward fill for weather if needed

# ----------------------------------------------------------
# 4️⃣ Extract Time-based Features
# ----------------------------------------------------------
df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["dow"] = df["datetime"].dt.weekday
df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

# ----------------------------------------------------------
# 5️⃣ Encode Weather Column
# ----------------------------------------------------------
le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"].astype(str))

# ----------------------------------------------------------
# 6️⃣ Drop unnecessary columns
# ----------------------------------------------------------
df = df.drop(columns=["datetime"], errors="ignore")

# ----------------------------------------------------------
# 7️⃣ Define Features (X) and Target (y)
# ----------------------------------------------------------
target = "pm2_5"
X = df.drop(columns=[target])
y = df[target]

# ----------------------------------------------------------
# 8️⃣ Split Data
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
# 9️⃣ Scale Numeric Features
# ----------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 🔟 Model 1: Random Forest
# ----------------------------------------------------------
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ----------------------------------------------------------
# 11️⃣ Model 2: Ridge Regression
# ----------------------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# ----------------------------------------------------------
# 12️⃣ Model 3: Neural Network
# ----------------------------------------------------------
# nn = Sequential([
#     Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
#     Dense(32, activation="relu"),
#     Dense(1)
# ])
# nn.compile(optimizer="adam", loss="mse")
# nn.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)
# y_pred_nn = nn.predict(X_test_scaled).flatten()

# ----------------------------------------------------------
# 13️⃣ Evaluate Models
# ----------------------------------------------------------
# ----------------------------------------------------------
# 13️⃣ Evaluate Models
# ----------------------------------------------------------
def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # ✅ Custom Accuracy (based on mean absolute percentage)
    accuracy = 100 * (1 - (mae / np.mean(y_true)))
    accuracy = max(0, min(accuracy, 100))  # Clamp between 0–100
    
    print(f"\n📊 {name} Performance:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "accuracy": accuracy}

results = [
    {**evaluate_model(y_test, y_pred_rf, "Random Forest"), "model": rf},
    {**evaluate_model(y_test, y_pred_ridge, "Ridge Regression"), "model": ridge},
    # {**evaluate_model(y_test, y_pred_nn, "Neural Network"), "model": nn}
]

# ✅ Select best model by Accuracy
best_result = max(results, key=lambda x: x["accuracy"])
best_model_name = best_result["name"]
best_trained_model = best_result["model"]

print(f"\n🏆 Best model selected based on Accuracy: {best_model_name}")

# ----------------------------------------------------------
# 15️⃣ Register Model in Hopsworks
# ----------------------------------------------------------
mr = project.get_model_registry()

# Choose correct filename
if best_model_name == "Neural Network":
    model_file = "best_aqi_model_nn.h5"
else:
    model_file = "best_aqi_model.pkl"

# ✅ Save model locally
joblib.dump(best_trained_model, model_file)

# ✅ Register model with metadata
model_meta = mr.python.create_model(
    name="aqi_forecast_model",
    metrics={
        "rmse": best_result["rmse"],
        "mae": best_result["mae"],
        "r2": best_result["r2"],
        "accuracy": best_result["accuracy"]
    },
    input_example=X_train.iloc[:1],
    description=f"Best model: {best_model_name} trained for AQI prediction (with accuracy tracking)"
)

# ✅ Save the model to Hopsworks
model_meta.save(model_file)

print("✅ Model successfully stored in Hopsworks Model Registry with accuracy metric.")