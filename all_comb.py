########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect human bodies and draw their modelled skeleton
   in an OpenGL window. It optionally triggers three different scenarios based on
   command-line flags:
   ... (truncated for brevity)
"""

import cv2
import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import threading
import tkinter as tk
from tkinter import ttk

# === HVAC Scenario Setup ===
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
WEATHERAPI_KEY = "cd11e7acf0c444ab876132252251304"

# Load HVAC model and encoder
model = joblib.load("./hvac/hvac_model.pkl")
le    = joblib.load("./hvac/hvac_label_encoder.pkl")

# Functions for HVAC scenario

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

# Global dictionary to store dashboard information (shared with Tkinter thread)
dashboard_data = {
    "attendance": "N/A",
    "num_bodies": 0,
    "alerts": [],
    "report": "",
    "hvac_insight": "N/A"
}

# -------------------- Tkinter Dashboard Functions --------------------
def update_dashboard_labels(root, attendance_label, bodies_label,
                             alerts_label, hvac_label, report_text):
    attendance_label.config(text="Attendance: " + dashboard_data["attendance"])
    bodies_label.config(text="Bodies detected: " + str(dashboard_data["num_bodies"]))
    alerts_label.config(text="Alerts: " +
                         (", ".join(dashboard_data["alerts"]) if dashboard_data["alerts"] else "None"))
    hvac_label.config(text="HVAC: " + dashboard_data["hvac_insight"])

    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, dashboard_data["report"])
    root.after(1000, update_dashboard_labels,
               root, attendance_label, bodies_label, alerts_label, hvac_label, report_text)


def run_dashboard():
    root = tk.Tk()
    root.title("Lecture Attendance Dashboard")
    root.geometry("600x450")
    root.configure(bg="#2D2D2D")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#2D2D2D", foreground="white")
    style.configure("Header.TLabel", font=("Helvetica", 18, "bold"),
                    background="#2D2D2D", foreground="#00FF7F")
    style.configure("Metric.TLabel", font=("Helvetica", 16),
                    background="#2D2D2D", foreground="white")
    style.configure("Report.TLabel", font=("Helvetica", 14),
                    background="#2D2D2D", foreground="white")

    header_label = ttk.Label(root, text="Lecture Attendance Dashboard",
                             style="Header.TLabel")
    header_label.pack(pady=(10, 20))

    metrics_frame = ttk.Frame(root)
    metrics_frame.pack(pady=10, fill="x", padx=20)

    attendance_label = ttk.Label(metrics_frame, text="Attendance: N/A",
                                 style="Metric.TLabel")
    attendance_label.pack(pady=5, anchor="w")

    bodies_label = ttk.Label(metrics_frame, text="Bodies detected: 0",
                              style="Metric.TLabel")
    bodies_label.pack(pady=5, anchor="w")

    alerts_label = ttk.Label(metrics_frame, text="Alerts: None",
                              style="Metric.TLabel")
    alerts_label.pack(pady=5, anchor="w")

    hvac_label = ttk.Label(metrics_frame, text="HVAC: N/A",
                            style="Metric.TLabel")
    hvac_label.pack(pady=5, anchor="w")

    report_frame = ttk.LabelFrame(root, text="Detailed Report",
                                  style="TLabel")
    report_frame.pack(padx=20, pady=20, fill="both", expand=True)

    report_text = tk.Text(report_frame, wrap="word", font=("Helvetica", 12),
                          bg="#1C1C1C", fg="white", relief="flat")
    report_text.pack(expand=True, fill="both", padx=10, pady=10)

    root.after(1000, update_dashboard_labels,
               root, attendance_label, bodies_label, alerts_label, hvac_label, report_text)
    root.mainloop()

# Start dashboard in a separate background thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# --------------------- ZED Tracking Code ---------------------
# (unchanged parse_args and main setup...)

def parse_args(init):
    # ... existing implementation unchanged
    pass  # truncated for brevity


def main(registered_students=None, covid_bodies=12):
    print("Running Body Tracking sample ...")
    print("Press 'q' to quit, 'm' to pause/restart, 'r' to report attendance, 'h' for HVAC update")

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    parse_args(init_params)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        exit(1)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True
    body_param.enable_body_fitting = False
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_18
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(
        min(camera_info.camera_configuration.resolution.width, 1280),
        min(camera_info.camera_configuration.resolution.height, 720)
    )
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width,
                   display_resolution.height / camera_info.camera_configuration.resolution.height]

    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam,
                body_param.enable_tracking, body_param.body_format)
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10

    if opt.attendance:
        attendance_max = 0
        attendance_min = float('inf')
        tracked_bodies = {}

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)

            alerts = []
            # ... existing scenarios unchanged

            # Update shared dashboard data
            dashboard_data["num_bodies"] = len(bodies.body_list)
            dashboard_data["alerts"] = alerts

            # Render views
            viewer.update_view(image, bodies)
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale,
                                bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)

            key = cv2.waitKey(key_wait)
            if key == ord('q'):
                print("Exiting...")
                break
            if key == ord('m'):
                key_wait = 0 if key_wait > 0 else 10
            if opt.attendance and key == ord('r'):
                # ... existing report generation unchanged
                pass

            # HVAC update on 'h'
            if key == ord('h'):
                try:
                    token = get_access_token()
                    # fetch live sensor data
                    sensor_data = {}
                    for var in VARIABLES:
                        val = get_variable_value(token, THING_ID, var)
                        sensor_data[var.lower()] = val
                    external_temp, local_time = get_weather_and_time()
                    occupancy = len(bodies.body_list)

                    # build input and predict
                    sample = [[
                        sensor_data['temperature'], sensor_data['humidity'],
                        sensor_data['sound_level'], sensor_data['lightlevel'],
                        sensor_data['airquality'], external_temp, occupancy
                    ]]
                    pred = model.predict(sample)
                    action = le.inverse_transform(pred)[0]

                    # suggest adjustment
                    target = 25.0
                    curr = sensor_data['temperature']
                    diff = curr - target
                    contrib = occupancy * 0.3
                    adj = round(diff + contrib, 1)
                    mag = abs(adj)
                    if action == "COOL" and adj > 0:
                        insight = f"❄️ COOL by {mag}°C to reach {target}°C (incl. {occupancy} ppl)"
                    elif action == "HEAT" and adj < 0:
                        insight = f"🔥 HEAT by {mag}°C to reach {target}°C (incl. {occupancy} ppl)"
                    elif action == "MAINTAIN":
                        insight = "✅ Maintain – temperature is optimal."
                    elif action == "FAN":
                        insight = "🌀 Run fan – circulate air."
                    elif action == "IDLE":
                        insight = "💤 Idle – no one’s here."
                    else:
                        insight = "ℹ️ Monitor – no immediate action."

                    dashboard_data['hvac_insight'] = insight
                    print(f"[HVAC] {insight}")
                except Exception as e:
                    print(f"HVAC update failed: {e}")

    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... existing arguments unchanged
    opt = parser.parse_args()
    # ... attendance input unchanged
    main(registered_students, covid_bodies)
