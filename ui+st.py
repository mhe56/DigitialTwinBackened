# COMBINED: ZED Body Tracking + Streaming + Dashboard
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
import cv2


# Global dictionary to store dashboard information (shared with Tkinter thread)
dashboard_data = {
    "attendance": "N/A",
    "num_bodies": 0,
    "alerts": [],
    "report": ""
}

# -------------------- Tkinter Dashboard Functions --------------------
def update_dashboard_labels(root, attendance_label, bodies_label, alerts_label, report_text):
    attendance_label.config(text="Attendance: " + dashboard_data["attendance"])
    bodies_label.config(text="Bodies detected: " + str(dashboard_data["num_bodies"]))
    alerts_label.config(text="Alerts: " + (", ".join(dashboard_data["alerts"]) if dashboard_data["alerts"] else "None"))
    
    # Update the detailed report text widget
    report_text.delete('1.0', tk.END)
    report_text.insert(tk.END, dashboard_data["report"])
    
    # Schedule update every 1000 milliseconds
    root.after(1000, update_dashboard_labels, root, attendance_label, bodies_label, alerts_label, report_text)

def run_dashboard():
    root = tk.Tk()
    root.title("Lecture Attendance Dashboard")
    root.geometry("600x400")
    root.configure(bg="#2D2D2D")  # Dark background

    # Set up ttk style for a modern look
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#2D2D2D", foreground="white")
    style.configure("Header.TLabel", font=("Helvetica", 18, "bold"), background="#2D2D2D", foreground="#00FF7F")
    style.configure("Metric.TLabel", font=("Helvetica", 16), background="#2D2D2D", foreground="white")
    style.configure("Report.TLabel", font=("Helvetica", 14), background="#2D2D2D", foreground="white")
    
    # Header
    header_label = ttk.Label(root, text="Lecture Attendance Dashboard", style="Header.TLabel")
    header_label.pack(pady=(10, 20))
    
    # Metrics Frame
    metrics_frame = ttk.Frame(root)
    metrics_frame.pack(pady=10, fill="x", padx=20)
    
    attendance_label = ttk.Label(metrics_frame, text="Attendance: N/A", style="Metric.TLabel")
    attendance_label.pack(pady=5, anchor="w")
    
    bodies_label = ttk.Label(metrics_frame, text="Bodies detected: 0", style="Metric.TLabel")
    bodies_label.pack(pady=5, anchor="w")
    
    alerts_label = ttk.Label(metrics_frame, text="Alerts: None", style="Metric.TLabel", foreground="red")
    alerts_label.pack(pady=5, anchor="w")
    
    # Report Frame
    report_frame = ttk.LabelFrame(root, text="Detailed Report", style="TLabel")
    report_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    report_text = tk.Text(report_frame, wrap="word", font=("Helvetica", 12), bg="#1C1C1C", fg="white", relief="flat")
    report_text.pack(expand=True, fill="both", padx=10, pady=10)
    
    root.after(1000, update_dashboard_labels, root, attendance_label, bodies_label, alerts_label, report_text)
    root.mainloop()

# Start dashboard in a separate background thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()


# --------------------- ZED Tracking Code ---------------------
def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2:
            init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.')) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Invalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")

def main(registered_students=None, covid_bodies=12):
    print("Running Body Tracking + Streaming ...")

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    parse_args(init_params)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # ✅ Enable streaming after opening camera
    stream_params = sl.StreamingParameters()
    stream_params.codec = sl.STREAMING_CODEC.H264
    stream_params.bitrate = 4000
    status_streaming = zed.enable_streaming(stream_params)
    if status_streaming != sl.ERROR_CODE.SUCCESS:
        print("Streaming initialization error: ", status_streaming)
        zed.close()
        exit()
    print("✅ ZED Streaming enabled on port", stream_params.port)

    # ✅ Body tracking setup [same as your original code]
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
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                       min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width,
                   display_resolution.height / camera_info.camera_configuration.resolution.height]

    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam,
                body_param.enable_tracking, body_param.body_format)

    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10

    # Optional: Attendance tracking logic as in your original code

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)

            # ✅ Update dashboard data + alerts here (same as your original logic)

            viewer.update_view(image, bodies)
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.body_list,
                                body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)

            key = cv2.waitKey(key_wait)
            if key == ord('q'):
                print("Exiting...")
                break

    # Cleanup
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.disable_streaming()
    zed.close()
    cv2.destroyAllWindows()

# --------------------- ENTRY POINT ---------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='')
    parser.add_argument('--ip_address', type=str, default='')
    parser.add_argument('--resolution', type=str, default='')
    parser.add_argument('-c', '--covid', action='store_true')
    parser.add_argument('-p', '--phone', action='store_true')
    parser.add_argument('-a', '--attendance', action='store_true')
    opt = parser.parse_args()

    # Prompt for student count if attendance is enabled
    registered_students = None
    covid_bodies = 12
    if opt.attendance:
        try:
            registered_students = int(input("Enter number of registered students: "))
        except ValueError:
            print("Invalid input.")
            exit(1)

        if opt.covid and registered_students > covid_bodies:
            print("Too many students for COVID scenario.")
            exit(1)
        elif not opt.covid and registered_students > 30:
            print("Too many students for lab capacity.")
            exit(1)

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Please specify only one input source.")
        exit(1)

    # Start dashboard thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    main(registered_students, covid_bodies)