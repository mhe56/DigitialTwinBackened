import sys
import numpy as _np
import numpy.core as _ncore
import time
from functools import lru_cache

# === Shim for numpy._core when unpickling old models ===
sys.modules['numpy._core'] = _ncore
sys.modules['numpy._core.numeric']    = _ncore.numeric
sys.modules['numpy._core.multiarray'] = _ncore.multiarray

# === Suppress scikit‑learn version‑mismatch warning ===
from sklearn.exceptions import InconsistentVersionWarning
import warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

import requests
import joblib
import numpy as np

# === Arduino Cloud API Setup ===
CLIENT_ID     = "gbx8Qihkr1iDPHb5MwCMgNx1MGM1G4tT"
CLIENT_SECRET = "0H31jZZyeqnMdgtWxqovUF008leXsOT2fX1sEAetYBE9q4MvaJnyXi5Mi6khdS5t"
THING_ID      = "0b50e8b3-d306-4e04-a163-c5d40382aec0"
VARIABLES     = ["Temperature", "Humidity", "Sound_Level", "airquality", "lightlevel"]

# === WeatherAPI.com Setup ===
WEATHERAPI_KEY = "cd11e7acf0c444ab876132252251304"

# === Model and encoder loaded once ===
_model = joblib.load("./hvac/hvac_model.pkl")
_label_encoder = joblib.load("./hvac/hvac_label_encoder.pkl")

# Cache for sensor data and weather data
_sensor_cache = {"timestamp": 0, "data": None}
_weather_cache = {"timestamp": 0, "data": None}
CACHE_DURATION = 60  # Cache duration in seconds

@lru_cache(maxsize=32)
def get_access_token():
    """
    Obtain an access token from Arduino Cloud API.
    """
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

def get_cached_sensor_data():
    """
    Get sensor data with caching
    """
    current_time = time.time()
    if current_time - _sensor_cache["timestamp"] < CACHE_DURATION and _sensor_cache["data"]:
        return _sensor_cache["data"]
    
    try:
        token = get_access_token()
        sensor_data = {}
        for var in VARIABLES:
            val = get_variable_value(token, THING_ID, var)
            sensor_data[var.lower()] = val
        
        _sensor_cache["data"] = sensor_data
        _sensor_cache["timestamp"] = current_time
        return sensor_data
    except Exception as e:
        print(f"Error fetching sensor data: {e}")
        return _sensor_cache["data"] or {var.lower(): 0 for var in VARIABLES}

def get_cached_weather_data(city="Beirut"):
    """
    Get weather data with caching
    """
    current_time = time.time()
    if current_time - _weather_cache["timestamp"] < CACHE_DURATION and _weather_cache["data"]:
        return _weather_cache["data"]
    
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={city}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        weather_data = {
            "temp": data["current"]["temp_c"],
            "time": data["location"]["localtime"]
        }
        _weather_cache["data"] = weather_data
        _weather_cache["timestamp"] = current_time
        return weather_data
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return _weather_cache["data"] or {"temp": 25, "time": "N/A"}

def get_variable_value(token, thing_id, name):
    """
    Fetch the latest property value for a given variable name.
    """
    url = f"https://api2.arduino.cc/iot/v2/things/{thing_id}/properties"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    for prop in resp.json():
        if prop.get("name") == name:
            return float(prop.get("last_value", 0))
    return None

def get_weather_and_time(city="Beirut", api_key=WEATHERAPI_KEY):
    """
    Fetch current temperature (°C) and local time for a given city.
    """
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data["current"]["temp_c"], data["location"]["localtime"]

def _compute_adjusted_diff(current_temp, occupancy, target_temp=25.0, heat_per_person=0.3):
    """
    Compute the temperature adjustment needed, including occupancy effect.
    """
    raw_diff = current_temp - target_temp
    occupancy_contrib = occupancy * heat_per_person
    adjusted = round(raw_diff + occupancy_contrib, 1)
    return adjusted

def predict_hvac_action(occupancy, city="Beirut"):
    """
    Predict the HVAC action and generate adjustment suggestion.
    """
    try:
        # Get cached sensor and weather data
        sensor_data = get_cached_sensor_data()
        weather_data = get_cached_weather_data(city)
        
        # Build input for model
        sample = np.array([[
            sensor_data.get("temperature", 25),
            sensor_data.get("humidity", 50),
            sensor_data.get("sound_level", 50),
            sensor_data.get("lightlevel", 50),
            sensor_data.get("airquality", 50),
            weather_data["temp"],
            occupancy
        ]])

        # Predict
        pred = _model.predict(sample)
        action = _label_encoder.inverse_transform(pred)[0]

        # Suggest adjustment
        adj_diff = _compute_adjusted_diff(sensor_data.get("temperature", 25), occupancy)
        abs_diff = abs(adj_diff)
        
        if action == "COOL" and adj_diff > 0:
            suggestion = f"❄️ COOL by {abs_diff}°C to reach 25°C (incl. {occupancy} ppl)."
        elif action == "HEAT" and adj_diff < 0:
            suggestion = f"🔥 HEAT by {abs_diff}°C to reach 25°C (incl. {occupancy} ppl)."
        elif action == "MAINTAIN":
            suggestion = "✅ Maintain – temperature is optimal."
        elif action == "FAN":
            suggestion = "🌀 Run fan – circulate air."
        elif action == "IDLE":
            suggestion = "💤 Idle – no one's here."
        else:
            suggestion = "ℹ️ Monitor – no immediate action."

        return {
            'action': action,
            'suggestion': suggestion,
            'sensor_data': sensor_data,
            'external_temp': weather_data["temp"],
            'local_time': weather_data["time"]
        }
    except Exception as e:
        print(f"Error in HVAC prediction: {e}")
        return {
            'action': 'ERROR',
            'suggestion': 'Error in HVAC prediction',
            'sensor_data': {},
            'external_temp': 25,
            'local_time': 'N/A'
        }

if __name__ == "__main__":  # Simple CLI usage
    try:
        occ = int(input("👥 Enter number of people in room (0–15): "))
    except ValueError:
        print("Please enter a valid integer for occupancy.")
        sys.exit(1)

    result = predict_hvac_action(occ)
    print("✅ HVAC Prediction & Suggestion:")
    print(f"Predicted Action: {result['action']}")
    print(result['suggestion'])
    print(f"Sensor Data: {result['sensor_data']}")
    print(f"External Temp: {result['external_temp']}°C")
    print(f"Local Time: {result['local_time']}")
