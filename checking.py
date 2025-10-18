import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

# Replace 'combined_data' with your actual feature group name
feature_group = fs.get_feature_group("testing_2", version=1)
df = feature_group.read()
print(df.tail(36))  # show last few rows


import requests, pandas as pd

# url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.9&longitude=67.0&hourly=pm10,pm2_5"
# data = requests.get(url).json()
# print(data["hourly"]["time"][-5:])
# print(data["hourly"]["pm10"][-5:])
