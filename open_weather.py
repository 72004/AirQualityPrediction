import requests
import time
import datetime
import pandas as pd

API_KEY = "6002ac178c1ae72e8625cd919f36f421"  # replace with your key
LAT, LON = 24.8607, 67.0011  # Karachi

all_data = []

print("📡 Fetching hourly weather data for the past 5 days...\n")

for days_ago in range(1, 21):
    print(f"📅 Day {days_ago} —", end=" ")
    base_time = int(time.time()) - days_ago * 86400

    for hour in range(24):
        dt = base_time - hour * 3600  # every past hour
        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        params = {
            "lat": LAT,
            "lon": LON,
            "dt": dt,
            "appid": API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200 and "data" in data:
            for d in data["data"]:
                record = {
                    "datetime": datetime.datetime.utcfromtimestamp(d["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "temp_C": d.get("temp"),
                    "feels_like_C": d.get("feels_like"),
                    "humidity_%": d.get("humidity"),
                    "pressure_hPa": d.get("pressure"),
                    "wind_speed_m/s": d.get("wind_speed"),
                    "clouds_%": d.get("clouds"),
                    "weather": d["weather"][0]["description"] if "weather" in d else None
                }
                all_data.append(record)

    print("✅ Hourly data added.")

df = pd.DataFrame(all_data)
df.to_csv("karachi_weather_5days_hourly.csv", index=False)

print(f"\n📊 Total records fetched: {len(df)}")
print("💾 Saved → karachi_weather_5days_hourly.csv")
