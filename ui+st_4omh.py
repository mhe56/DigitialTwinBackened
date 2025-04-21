#!/usr/bin/env python3
########################################################################
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
########################################################################

"""
Combined sample: ZED camera H264 streaming + human body tracking with OpenGL & Tkinter dashboard.
Usage:
  python zed_streaming_body_tracking.py [--input_svo_file SVO] [--ip_address IP[:port]] [--resolution RES] \
    [--covid] [--phone] [--attendance] [--stream]
Options:
  --stream        Enable H264 streaming on default port
  -c, --covid     COVID-19 distancing alerts
  -p, --phone     Phone usage detection
  -a, --attendance  Attendance classification
"""
import sys
import time
import argparse
import threading
import tkinter as tk
from tkinter import ttk
from time import sleep

import cv2
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

# -------------------- Global Dashboard Data --------------------
dashboard_data = {
    "attendance": "N/A",
    "num_bodies": 0,
    "alerts": [],
    "report": ""
}

# -------------------- Tkinter Dashboard --------------------
def update_dashboard_labels(root, attendance_label, bodies_label, alerts_label, report_text):
    attendance_label.config(text="Attendance: " + dashboard_data["attendance"])
    bodies_label.config(text="Bodies detected: " + str(dashboard_data["num_bodies"]))
    alerts_label.config(text="Alerts: " + (", ".join(dashboard_data["alerts"]) if dashboard_data["alerts"] else "None"))
    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, dashboard_data["report"])
    root.after(1000, update_dashboard_labels, root, attendance_label, bodies_label, alerts_label, report_text)


def run_dashboard():
    root = tk.Tk()
    root.title("Lecture Attendance Dashboard")
    root.geometry("600x400")
    root.configure(bg="#2D2D2D")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Header.TLabel", font=("Helvetica", 18, "bold"), background="#2D2D2D", foreground="#00FF7F")
    style.configure("Metric.TLabel", font=("Helvetica", 16), background="#2D2D2D", foreground="white")

    header = ttk.Label(root, text="Lecture Attendance Dashboard", style="Header.TLabel")
    header.pack(pady=(10,20))

    frame = ttk.Frame(root)
    frame.pack(fill="x", padx=20)
    attendance_lbl = ttk.Label(frame, text="Attendance: N/A", style="Metric.TLabel")
    attendance_lbl.pack(anchor="w", pady=5)
    bodies_lbl = ttk.Label(frame, text="Bodies detected: 0", style="Metric.TLabel")
    bodies_lbl.pack(anchor="w", pady=5)
    alerts_lbl = ttk.Label(frame, text="Alerts: None", style="Metric.TLabel")
    alerts_lbl.pack(anchor="w", pady=5)

    report_frame = ttk.LabelFrame(root, text="Detailed Report")
    report_frame.pack(fill="both", expand=True, padx=20, pady=20)
    report_txt = tk.Text(report_frame, wrap="word", bg="#1C1C1C", fg="white")
    report_txt.pack(fill="both", expand=True)

    root.after(1000, update_dashboard_labels, root, attendance_lbl, bodies_lbl, alerts_lbl, report_txt)
    root.mainloop()

# Launch dashboard thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# -------------------- Argument Parsing --------------------

def parse_args(init_params):
    # Resolution options
    if opt.resolution.upper() in ["HD2K","HD1200","HD1080","HD720","SVGA","VGA"]:
        init_params.camera_resolution = getattr(sl.RESOLUTION, opt.resolution.upper())
        print(f"Using camera resolution: {opt.resolution.upper()}")
    else:
        print("Using default resolution")

    # SVO / IP stream
    if opt.input_svo_file and opt.input_svo_file.endswith('.svo'):
        init_params.set_from_svo_file(opt.input_svo_file)
        print(f"Using SVO file: {opt.input_svo_file}")
    elif opt.ip_address:
        addr=opt.ip_address
        parts=addr.split(':')
        if len(parts)==2:
            init_params.set_from_stream(parts[0], int(parts[1]))
        else:
            init_params.set_from_stream(addr)
        print(f"Using stream IP: {opt.ip_address}")

# -------------------- Main --------------------

def main(registered_students=None, covid_bodies=12):
    print("Starting combined streaming & body tracking...")
    print("Press 'q' to quit, 'm' to pause/restart, 'r' for report (if attendance)")

    # Initialize camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    parse_args(init_params)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {err}")
        sys.exit(1)

    # Optional H264 streaming
    runtime = sl.RuntimeParameters()
    if opt.stream:
        stream_p = sl.StreamingParameters()
        stream_p.codec = sl.STREAMING_CODEC.H264
        stream_p.bitrate = 4000
        ret = zed.enable_streaming(stream_p)
        if ret != sl.ERROR_CODE.SUCCESS:
            print("Error enabling streaming", ret)
        else:
            print(f"Streaming enabled on port {stream_p.port}")

    # Positional & body tracking
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = False
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.body_format = sl.BODY_FORMAT.BODY_18
    zed.enable_body_tracking(body_params)
    body_rt = sl.BodyTrackingRuntimeParameters()
    body_rt.detection_confidence_threshold = 40

    # Viewer & frame setup
    cam_info = zed.get_camera_information()
    disp_res = sl.Resolution(min(cam_info.camera_configuration.resolution.width,1280),
                             min(cam_info.camera_configuration.resolution.height,720))
    image_scale = [disp_res.width/cam_info.camera_configuration.resolution.width,
                   disp_res.height/cam_info.camera_configuration.resolution.height]
    viewer = gl.GLViewer()
    viewer.init(cam_info.camera_configuration.calibration_parameters.left_cam,
                True, body_params.body_format)
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10
    attendance_max=0; attendance_min=float('inf'); tracked={}  # for attendance

    while viewer.is_available():
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, disp_res)
            zed.retrieve_bodies(bodies, body_rt)
            alerts=[]
            num = len(bodies.body_list)
            # COVID scenario
            if opt.covid:
                if num>covid_bodies: alerts.append(f"> {covid_bodies} bodies detected")
                for i in range(num):
                    for j in range(i+1,num):
                        if np.linalg.norm(bodies.body_list[i].position - bodies.body_list[j].position)<1.0:
                            alerts.append("Two bodies <1m apart")
            # Phone detection
            if opt.phone and bodies.body_list:
                kp = bodies.body_list[0].keypoint_2d
                if len(kp)>1 and abs(kp[0][1]-kp[1][1])<40:
                    alerts.append("Phone usage alert")
            # Attendance
            if opt.attendance and registered_students is not None:
                count=num; attendance_max=max(attendance_max,count); attendance_min=min(attendance_min,count)
                now=time.time()
                for b in bodies.body_list:
                    bid=int(b.id) if hasattr(b.id,'__int__') else hash(b)
                    tracked.setdefault(bid,{'first':now,'last':now})['last']=now
                ratio=attendance_max/registered_students
                cls="Poor" if ratio<(1/3) else "Fair" if ratio<=(2/3) else "Good"
                dashboard_data["attendance"]=f"{cls} ({count}/{registered_students})"
            dashboard_data["num_bodies"]=num
            dashboard_data["alerts"]=alerts
            # Render
            viewer.update_view(image,bodies)
            img2d=image.get_data()
            cv_viewer.render_2D(img2d,image_scale,bodies.body_list,True,body_params.body_format)
            cv2.imshow("ZED | 2D View",img2d)
            key=cv2.waitKey(key_wait)
            if key==ord('q'): break
            if key==ord('m'): key_wait = 0 if key_wait>0 else 10
            if opt.attendance and key==ord('r'):
                lines=["--- Attendance Report ---",
                       f"Max: {attendance_max}", f"Min: {attendance_min}"]
                for bid,t in tracked.items():
                    lines.append(f"ID {bid}: {t['last']-t['first']:.2f}s")
                lines.append("-----------------------")
                dashboard_data["report"] = "\n".join(lines)
                print(dashboard_data["report"])

    # Cleanup
    if opt.stream: zed.disable_streaming()
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='', help='Path to .svo file')
    parser.add_argument('--ip_address', type=str, default='', help='IP for stream')
    parser.add_argument('--resolution', type=str, default='', help='Camera resolution')
    parser.add_argument('-c', '--covid', action='store_true', help='COVID distancing')
    parser.add_argument('-p', '--phone', action='store_true', help='Phone detect')
    parser.add_argument('-a', '--attendance', action='store_true', help='Attendance mode')
    parser.add_argument('--stream', action='store_true', help='Enable H264 streaming')
    opt = parser.parse_args()

    reg_students=None; covid_bodies=12
    if opt.attendance:
        try:
            reg_students=int(input("Enter registered student count: "))
        except ValueError:
            print("Invalid number"); sys.exit(1)
        if opt.covid and reg_students>covid_bodies:
            print("Error: max for COVID is 12"); sys.exit(1)
        if not opt.covid and reg_students>30:
            print("Error: max capacity is 30"); sys.exit(1)

    main(reg_students, covid_bodies)
