import hopsworks
import os

# === Step 1: Login ===
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))

# === Step 2: Access the Model Registry ===
mr = project.get_model_registry()

# === Step 3: Automatically get the latest version of your model ===
model = mr.get_model("AQI_RF_Forecaster", version=1)  # None = latest version

# === Step 4: Download locally ===
local_dir = "models/AQI_RF_Forecaster_latest"
os.makedirs(local_dir, exist_ok=True)

model_dir = model.download(local_dir)
print(f"✅ Model downloaded successfully to: {model_dir}")
