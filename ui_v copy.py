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
   in an OpenGL window, with HVAC prediction on 'h' key press.
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
from predict_hvac import predict_hvac_action  # renamed from run_hvac_prediction

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
    print("Running Body Tracking sample ...")
    print("Press 'q' to quit, 'm' to pause/restart, 'r' for report, 'h' for HVAC suggestion")

    # Create and configure camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    parse_args(init_params)
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable tracking
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    bt_param = sl.BodyTrackingParameters()
    bt_param.enable_tracking = True
    bt_param.enable_body_fitting = False
    bt_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    bt_param.body_format = sl.BODY_FORMAT.BODY_18
    zed.enable_body_tracking(bt_param)
    runtime_param = sl.BodyTrackingRuntimeParameters()
    runtime_param.detection_confidence_threshold = 40

    camera_info = zed.get_camera_information()
    display_res = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_res.width / camera_info.camera_configuration.resolution.width,
                   display_res.height / camera_info.camera_configuration.resolution.height]

    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam,
                bt_param.enable_tracking, bt_param.body_format)
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10
    frame_counter = 0

    # Attendance tracking variables
    if opt.attendance:
        attendance_max = 0
        attendance_min = float('inf')
        tracked_bodies = {}

    while viewer.is_available():
        frame_counter += 1
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            zed.retrieve_bodies(bodies, runtime_param)

            alerts = []
            num_bodies = len(bodies.body_list)

# ---------------- COVID-19 Scenario ----------------
            if opt.covid:
                num_bodies = len(bodies.body_list)
                if num_bodies > covid_bodies:
                    alerts.append("More than {} bodies detected!".format(covid_bodies))
                for i in range(num_bodies):
                    for j in range(i + 1, num_bodies):
                        pos1 = bodies.body_list[i].position
                        pos2 = bodies.body_list[j].position
                        if np.linalg.norm(pos1 - pos2) < 1.0:
                            alerts.append("Two bodies < 1m apart!")
            # -----------------------------------------------------
            
            # ---------------- Phone Detection Scenario ----------------
            if opt.phone:
                if bodies.is_new:
                    body_array = bodies.body_list
                    if body_array:
                        first_body = body_array[0]
                        keypoint_2d = first_body.keypoint_2d
                        if len(keypoint_2d) > 1:
                            try:
                                nose_y = keypoint_2d[0][1]
                                neck_y = keypoint_2d[1][1]
                            except Exception:
                                try:
                                    nose_y = keypoint_2d[0].y
                                    neck_y = keypoint_2d[1].y
                                except Exception:
                                    nose_y = neck_y = None
                            if nose_y is not None and neck_y is not None:
                                threshold = 40
                                if abs(nose_y - neck_y) < threshold:
                                    alerts.append("Phone usage alert!")
            # -----------------------------------------------------------
            
            # ---------------- Attendance Classification Scenario ----------------
            if opt.attendance and registered_students is not None:
                present_count = len(bodies.body_list)
                attendance_max = max(attendance_max, present_count)
                attendance_min = min(attendance_min, present_count)
                current_time = time.time()
                for body in bodies.body_list:
                    try:
                        body_id = int(body.id)
                    except Exception:
                        body_id = hash(body)
                    if body_id not in tracked_bodies:
                        tracked_bodies[body_id] = {'first': current_time, 'last': current_time}
                    else:
                        tracked_bodies[body_id]['last'] = current_time
                ratio = attendance_max / registered_students
                if ratio < (1/3):
                    classification = "Poor"
                elif ratio <= (2/3):
                    classification = "Fair"
                else:
                    classification = "Good"
                dashboard_data["attendance"] = "{} ({} / {})".format(classification, present_count, registered_students)
            # ---------------------------------------------------------------------
        
            


            # Update shared dashboard data
            dashboard_data["num_bodies"] = num_bodies
            dashboard_data["alerts"] = alerts

            # Render views
            viewer.update_view(image, bodies)
            img_data = image.get_data()
            cv_viewer.render_2D(img_data, image_scale, bodies.body_list,
                                bt_param.enable_tracking, bt_param.body_format)
            cv2.imshow("ZED | 2D View", img_data)

            key = cv2.waitKey(key_wait)
            if key == ord('q'):
                print("Exiting...")
                break
            if key == ord('m'):
                key_wait = 0 if key_wait > 0 else 10

            # ---------------- Generate Attendance Report on 'r' ----------------
            if opt.attendance and key == ord('r'):
                report_lines = []
                report_lines.append("----- Lecture Attendance Tracking Report -----")
                report_lines.append("Maximum attendee count: {} students".format(attendance_max))
                report_lines.append("Minimum attendee count: {} students".format(attendance_min))
                for body_id, times in tracked_bodies.items():
                    duration = times['last'] - times['first']
                    report_lines.append("Body ID {}: tracked for {:.2f} seconds".format(body_id, duration))
                report_lines.append("----- End of Report -----")
                report_text = "\n".join(report_lines)
                dashboard_data["report"] = report_text
                print("\n" + report_text + "\n")
            # --------------------------------------------------------------------------------
                        # === HVAC suggestion on 'h' ===
            if key == ord('h'):
                # Call the HVAC predictor using number of detected bodies as occupancy
                hvac_result = predict_hvac_action(num_bodies)
                suggestion = hvac_result.get('suggestion', 'No suggestion')
                print(f"🔔 HVAC Suggestion ({num_bodies} occupants): {suggestion}")
                # Optionally append to dashboard alerts
                dashboard_data['alerts'].append(f"HVAC: {suggestion}")
        # End of while loop


    # Cleanup
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--ip_address', type=str, help='IP Address in format a.b.c.d:port or a.b.c.d, for streaming setups', default='')
    parser.add_argument('--resolution', type=str, help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, or VGA', default='')
    parser.add_argument('-c', '--covid', action='store_true', help='Enable COVID-19 scenario alerts')
    parser.add_argument('-p', '--phone', action='store_true', help='Enable phone detection scenario')
    parser.add_argument('-a', '--attendance', action='store_true', help='Enable attendance classification scenario')
    opt = parser.parse_args()
    
    registered_students = None
    covid_bodies = 12
    if opt.attendance:
        try:
            registered_students = int(input("Enter number of registered students: "))
        except ValueError:
            print("Invalid input. Please enter an integer.")
            exit(1)
        if opt.covid and registered_students > covid_bodies:
            print("Error: Max registered students exceeded for COVID scenario.")
            exit(1)
        if not opt.covid and registered_students > 30:
            print("Error: Lab capacity exceeded (30).")
            exit(1)
    if opt.input_svo_file and opt.ip_address:
        print("Specify only one input source.")
        exit(1)
    main(registered_students, covid_bodies)
