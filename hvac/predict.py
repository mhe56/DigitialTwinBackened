import sys
import numpy as _np
import numpy.core as _ncore

# === Shim for numpy._core when unpickling old models ===
sys.modules['numpy._core'] = _ncore
sys.modules['numpy._core.numeric']    = _ncore.numeric
sys.modules['numpy._core.multiarray'] = _ncore.multiarray

# === Suppress scikit‑learn version‐mismatch warning ===
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

import requests
import pandas as pd
import joblib
from datetime import datetime

# === Arduino Cloud API Setup ===
CLIENT_ID     = "gbx8Qihkr1iDPHb5MwCMgNx1MGM1G4tT"
CLIENT_SECRET = "0H31jZZyeqnMdgtWxqovUF008leXsOT2fX1sEAetYBE9q4MvaJnyXi5Mi6khdS5t"
THING_ID      = "0b50e8b3-d306-4e04-a163-c5d40382aec0"
VARIABLES     = ["Temperature", "Humidity", "Sound_Level", "airquality", "lightlevel"]

# === WeatherAPI.com Setup ===
WEATHERAPI_KEY = "cd11e7acf0c444ab876132252251304"  # ← Replace with your own key

def get_access_token():
    url = "https://api2.arduino.cc/iot/v1/clients/token"
    payload = {
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience":      "https://api2.arduino.cc/iot"
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_variable_value(token, thing_id, name):
    url = f"https://api2.arduino.cc/iot/v2/things/{thing_id}/properties"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    for prop in resp.json():
        if prop.get("name") == name:
            return float(prop.get("last_value", 0))
    return None

def get_weather_and_time(city="Beirut", api_key=WEATHERAPI_KEY):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data["current"]["temp_c"], data["location"]["localtime"]

# === Load model + label‐encoder ===
model = joblib.load("hvac_model.pkl")
le    = joblib.load("hvac_label_encoder.pkl")

# === Fetch live sensor data ===
token = get_access_token()
print("✅ Access token retrieved. Fetching sensor values...\n")

sensor_data = {}
for var in VARIABLES:
    val = get_variable_value(token, THING_ID, var)
    print(f"{var}: {val}")
    sensor_data[var.lower()] = val

# === External weather & local time ===
external_temp, local_time = get_weather_and_time()
print(f"\n🌤️ External Temp in Beirut: {external_temp}°C")
print(f"🕒 Local Time: {local_time}")

# === Occupancy ===
occupancy = int(input("👥 Enter number of people in room (0–15): "))

# === Build input vector ===
sample_input = [[
    sensor_data["temperature"],
    sensor_data["humidity"],
    sensor_data["sound_level"],
    sensor_data["lightlevel"],
    sensor_data["airquality"],
    external_temp,
    occupancy
]]

# === Predict ===
pred = model.predict(sample_input)
pred_action = le.inverse_transform(pred)[0]
print(f"\n🔮 Predicted HVAC Action: {pred_action}")

# === Suggest adjustment ===
target_temp = 25.0
current_temp = sensor_data["temperature"]
raw_diff = current_temp - target_temp

heat_per_person = 0.3
occupancy_contrib = occupancy * heat_per_person

adjusted_diff = raw_diff + occupancy_contrib
adjusted_diff = round(adjusted_diff, 1)
abs_diff = abs(adjusted_diff)

if pred_action == "COOL" and adjusted_diff > 0:
    print(f"❄️ COOL by {abs_diff}°C to reach {target_temp}°C (incl. {occupancy} ppl).")
elif pred_action == "HEAT" and adjusted_diff < 0:
    print(f"🔥 HEAT by {abs_diff}°C to reach {target_temp}°C (incl. {occupancy} ppl).")
elif pred_action == "MAINTAIN":
    print("✅ Maintain – temperature is optimal.")
elif pred_action == "FAN":
    print("🌀 Run fan – circulate air.")
elif pred_action == "IDLE":
    print("💤 Idle – no one’s here.")
else:
    print("ℹ️ Monitor – no immediate action.")
