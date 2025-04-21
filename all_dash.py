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
   
   - COVID-19 Scenario (-c / --covid):
       Alerts if more than 12 bodies are detected or if any two bodies are less
       than 1 meter apart.
       
   - Phone Detection Scenario (-p / --phone):
       Alerts if the nose and neck keypoints are too close, indicating that a student
       might be using their phone in class.
       
   - Attendance Classification (-a / --attendance):
       Tracks the number of students present compared to a registered student count.
       Categorizes attendance as:
           * Poor (< 1/3 attendance)
           * Fair (between 1/3 and 2/3 attendance)
           * Good (> 2/3 attendance)
           
       When attendance is enabled, the user is prompted for the registered student count.
       If both the COVID-19 and attendance scenarios are enabled, the registered student count
       must not exceed 12 (the maximum allowed for the COVID-19 scenario). Otherwise,
       when the COVID-19 scenario is not enabled, the registered student count must be <= 30.
       
       Additionally, while running, if the attendance scenario is enabled, pressing the 'r' key
       will generate a report of Lecture Attendance Tracking. This report includes:
           - The maximum and minimum attendee counts observed.
           - For each tracked student (body), how long they were tracked in the session.
       
   - HVAC Prediction (-h / --hvac):
       On pressing 'h', fetches and displays an HVAC adjustment suggestion based on
       current body count (occupancy). Runs asynchronously to avoid blocking.
       
   A Tkinter-based dashboard displays all this information with a modern look.
"""

import cv2
import sys
import time
import threading
import argparse
import tkinter as tk
from tkinter import ttk

import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

from predict_hvac import predict_hvac_action

# Shared data for dashboard (thread-safe enough for simple updates)
dashboard_data = {
    "attendance": "N/A",
    "num_bodies": 0,
    "alerts": [],
    "report": "",
    "hvac": "N/A"
}

# -------------------- Dashboard Functions --------------------

def update_dashboard_labels(root, attendance_label, bodies_label, alerts_label, hvac_label, report_text):
    attendance_label.config(text=f"Attendance: {dashboard_data['attendance']}")
    bodies_label.config(text=f"Bodies detected: {dashboard_data['num_bodies']}")
    alerts_label.config(text=f"Alerts: {', '.join(dashboard_data['alerts']) or 'None'}")
    hvac_label.config(text=f"HVAC: {dashboard_data['hvac']}")

    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, dashboard_data['report'])

    # Schedule next update
    root.after(1000, update_dashboard_labels, root,
               attendance_label, bodies_label, alerts_label, hvac_label, report_text)


def run_dashboard():
    root = tk.Tk()
    root.title("Lecture Attendance Dashboard")
    root.geometry("600x450")
    root.configure(bg="#2D2D2D")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#2D2D2D", foreground="white")
    style.configure("Header.TLabel", font=("Helvetica", 18, "bold"), foreground="#00FF7F")
    style.configure("Metric.TLabel", font=("Helvetica", 16), foreground="white")
    style.configure("Report.TLabel", font=("Helvetica", 14), foreground="white")

    header = ttk.Label(root, text="Lecture Attendance Dashboard", style="Header.TLabel")
    header.pack(pady=(10, 20))

    metrics = ttk.Frame(root)
    metrics.pack(padx=20, fill="x")

    attendance_label = ttk.Label(metrics, text="Attendance: N/A", style="Metric.TLabel")
    attendance_label.pack(anchor="w", pady=5)

    bodies_label = ttk.Label(metrics, text="Bodies detected: 0", style="Metric.TLabel")
    bodies_label.pack(anchor="w", pady=5)

    alerts_label = ttk.Label(metrics, text="Alerts: None", style="Metric.TLabel")
    alerts_label.pack(anchor="w", pady=5)

    hvac_label = ttk.Label(metrics, text="HVAC: N/A", style="Metric.TLabel")
    hvac_label.pack(anchor="w", pady=5)

    report_frame = ttk.LabelFrame(root, text="Detailed Report", style="TLabel")
    report_frame.pack(padx=20, pady=10, fill="both", expand=True)

    report_text = tk.Text(report_frame, wrap="word", font=("Helvetica", 12),
                          bg="#1C1C1C", fg="white", relief="flat")
    report_text.pack(fill="both", expand=True, padx=10, pady=10)

    # Kick off the periodic UI update
    root.after(1000, update_dashboard_labels, root,
               attendance_label, bodies_label, alerts_label, hvac_label, report_text)
    root.mainloop()

# Launch dashboard thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# --------------------- HVAC Fetch ---------------------

def fetch_and_update_hvac(occupancy):
    try:
        result = predict_hvac_action(occupancy)
        dashboard_data['hvac'] = result.get('suggestion', 'No suggestion')
    except Exception as e:
        dashboard_data['hvac'] = f"Error: {e}"

# --------------------- ZED Tracking Code ---------------------

def parse_args(init, opt):
    if opt.input_svo_file and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
    elif opt.ip_address:
        ip = opt.ip_address
        parts = ip.split(':')
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else None
        if port:
            init.set_from_stream(host, port)
        else:
            init.set_from_stream(host)
    # Resolution mapping
    resolutions = {
        'HD2K': sl.RESOLUTION.HD2K,
        'HD1200': sl.RESOLUTION.HD1200,
        'HD1080': sl.RESOLUTION.HD1080,
        'HD720': sl.RESOLUTION.HD720,
        'SVGA': sl.RESOLUTION.SVGA,
        'VGA': sl.RESOLUTION.VGA
    }
    if opt.resolution in resolutions:
        init.camera_resolution = resolutions[opt.resolution]


def main(registered_students=None, covid_bodies=12):
    print("Running Body Tracking sample ...")
    print("Press 'q' to quit, 'm' to pause/restart, 'r' for report, 'h' for HVAC update")

    # Initialize ZED
    zed = sl.Camera()
    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    parse_args(init, opt)
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        sys.exit(1)

    # Enable tracking
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    bt_params = sl.BodyTrackingParameters()
    bt_params.enable_tracking = True
    bt_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    bt_params.body_format = sl.BODY_FORMAT.BODY_18
    zed.enable_body_tracking(bt_params)
    rt_params = sl.BodyTrackingRuntimeParameters()
    rt_params.detection_confidence_threshold = 40

    info = zed.get_camera_information()
    disp_res = sl.Resolution(min(info.camera_configuration.resolution.width, 1280),
                             min(info.camera_configuration.resolution.height, 720))
    scale = [disp_res.width / info.camera_configuration.resolution.width,
             disp_res.height / info.camera_configuration.resolution.height]

    viewer = gl.GLViewer()
    viewer.init(info.camera_configuration.calibration_parameters.left_cam,
                bt_params.enable_tracking, bt_params.body_format)

    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10

    # Attendance tracking
    if opt.attendance:
        att_max = 0
        att_min = float('inf')
        tracked = {}

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, disp_res)
            zed.retrieve_bodies(bodies, rt_params)

            num = len(bodies.body_list)
            alerts = []

            if opt.covid:
                if num > covid_bodies:
                    alerts.append(f"More than {covid_bodies} bodies detected!")
                # pairwise distance check
                positions = [b.position for b in bodies.body_list]
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        if np.linalg.norm(positions[i] - positions[j]) < 1.0:
                            alerts.append("Two bodies < 1m apart!")

            if opt.phone and bodies.is_new and bodies.body_list:
                kp = bodies.body_list[0].keypoint_2d
                if len(kp) > 1:
                    nose_y = getattr(kp[0], 'y', kp[0][1])
                    neck_y = getattr(kp[1], 'y', kp[1][1])
                    if abs(nose_y - neck_y) < 40:
                        alerts.append("Phone usage alert!")

            if opt.attendance and registered_students is not None:
                present = num
                att_max = max(att_max, present)
                att_min = min(att_min, present)
                now = time.time()
                for b in bodies.body_list:
                    bid = int(b.id) if hasattr(b, 'id') else hash(b)
                    tracked.setdefault(bid, {'first': now, 'last': now})['last'] = now
                ratio = att_max / registered_students
                cls = "Poor" if ratio < 1/3 else "Fair" if ratio <= 2/3 else "Good"
                dashboard_data['attendance'] = f"{cls} ({present} / {registered_students})"

            dashboard_data['num_bodies'] = num
            dashboard_data['alerts'] = alerts

            viewer.update_view(image, bodies)
            frame = image.get_data()
            cv_viewer.render_2D(frame, scale, bodies.body_list,
                                bt_params.enable_tracking, bt_params.body_format)
            cv2.imshow("ZED | 2D View", frame)

            key = cv2.waitKey(key_wait)
            if key == ord('q'):
                break
            if key == ord('m'):
                key_wait = 0 if key_wait else 10
            if opt.attendance and key == ord('r'):
                lines = ["----- Lecture Attendance Tracking Report -----",
                         f"Max attendees: {att_max}",
                         f"Min attendees: {att_min}"]
                for bid, times in tracked.items():
                    dur = times['last'] - times['first']
                    lines.append(f"Body {bid}: {dur:.2f}s")
                lines.append("----- End of Report -----")
                dashboard_data['report'] = '\n'.join(lines)

            if key == ord('h'):
                threading.Thread(target=fetch_and_update_hvac,
                                 args=(num,), daemon=True).start()

    # Cleanup
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='')
    parser.add_argument('--ip_address', type=str, default='')
    parser.add_argument('--resolution', type=str, default='')
    parser.add_argument('-c', '--covid', action='store_true')
    parser.add_argument('-p', '--phone', action='store_true')
    parser.add_argument('-a', '--attendance', action='store_true')

    opt = parser.parse_args()
    registered_students = None
    covid_bodies = 12
    if opt.attendance:
        try:
            registered_students = int(input("Enter number of registered students: "))
        except ValueError:
            sys.exit("Please enter a valid integer.")
        if opt.covid and registered_students > covid_bodies:
            sys.exit("Error: exceeds COVID max.")
        if not opt.covid and registered_students > 30:
            sys.exit("Error: exceeds capacity.")
    main(registered_students, covid_bodies)
