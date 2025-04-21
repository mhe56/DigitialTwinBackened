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
   command‑line flags:
   
   - COVID‑19 Scenario (-c / --covid): 
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
       If both the COVID‑19 and attendance scenarios are enabled, the registered student count
       must not exceed 12 (the maximum allowed for the COVID‑19 scenario). Otherwise,
       when the COVID‑19 scenario is not enabled, the registered student count must be <= 30.
       
       Additionally, while running, if attendance is enabled, pressing the 'r' key will
       generate an attendance report.
       
   In this version a web‑based dashboard displays all the information in a modern graphical
   interface. Open a browser at http://localhost:5000 to view the dashboard.
"""

import cv2
import sys
import time
import threading
import argparse
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

# --------------------- Web Dashboard Imports ---------------------
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# ------------------- Global Shared Dashboard Data -------------------
dashboard_data = {
    "attendance": "N/A",
    "num_bodies": 0,
    "alerts": [],
    "report": ""
}

# --------------------- Flask Web Dashboard Setup ---------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# HTML template for the dashboard.
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Lecture Attendance Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.3.0/socket.io.js"></script>
    <script type="text/javascript">
      document.addEventListener("DOMContentLoaded", function(){
          var socket = io();
          socket.on("dashboard_update", function(data){
              document.getElementById("attendance").innerHTML = "Attendance: " + data.attendance;
              document.getElementById("num_bodies").innerHTML = "Bodies detected: " + data.num_bodies;
              if(data.alerts && data.alerts.length > 0){
                  document.getElementById("alerts").innerHTML = "Alerts: " + data.alerts.join(", ");
              } else {
                  document.getElementById("alerts").innerHTML = "Alerts: None";
              }
              document.getElementById("report").textContent = data.report;
          });
      });
    </script>
    <style>
      body { font-family: Helvetica, sans-serif; background-color: #2D2D2D; color: white; padding: 20px; }
      h1 { color: #00FF7F; }
      .metric { font-size: 24px; margin: 10px 0; }
      #report { background-color: #1C1C1C; padding: 10px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Lecture Attendance Dashboard</h1>
    <div id="attendance" class="metric">Attendance: N/A</div>
    <div id="num_bodies" class="metric">Bodies detected: 0</div>
    <div id="alerts" class="metric" style="color: red;">Alerts: None</div>
    <hr>
    <h2>Detailed Report</h2>
    <div id="report"></div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_template)

def dashboard_updater():
    """Background task: every second, emit the current dashboard_data to all clients."""
    while True:
        socketio.emit("dashboard_update", dashboard_data)
        time.sleep(1)

def start_web_server():
    # Start the background task that pushes dashboard updates.
    socketio.start_background_task(target=dashboard_updater)
    # Run the web server. (Adjust host/port as needed.)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)

# ------------------ ZED Tracking and Processing Code ------------------

# Use argparse to accept flags for scenarios.
parser = argparse.ArgumentParser()
parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file (if replaying)', default='')
parser.add_argument('--ip_address', type=str, help='IP Address in format a.b.c.d:port or a.b.c.d, for streaming', default='')
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
        print("Invalid input. Please enter an integer value for registered students.")
        exit(1)
    if opt.covid:
        if registered_students > covid_bodies:
            print("Error: COVID-19 scenario enabled. Maximum number of registered students exceeded!")
            exit(1)
    else:
        if registered_students > 30:
            print("Error: Lab capacity exceeded! Maximum registered students is 30.")
            exit(1)

if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
    print("Specify only input_svo_file or ip_address, or none to use a wired camera. Exiting.")
    exit(1)

def parse_tracking_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input:", opt.input_svo_file)
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4 and len(ip_str.split(":")) == 2:
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP:", ip_str)
        elif ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP:", ip_str)
        else:
            print("Invalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")

def main(registered_students=None, covid_bodies=12):
    print("Running Body Tracking sample ...")
    print("Press 'q' to quit, 'm' to pause/restart, 'r' to generate attendance report (if enabled)")
    
    # Create and open the ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    parse_tracking_args(init_params)
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        exit(1)
    
    # Enable positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # Set up body tracking
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True
    body_param.enable_body_fitting = False
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_18
    zed.enable_body_tracking(body_param)
    
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    
    # Get camera info & create viewer
    camera_info = zed.get_camera_information()
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                       min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width,
                   display_resolution.height / camera_info.camera_configuration.resolution.height]
    
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam,
                body_param.enable_tracking, body_param.body_format)
    
    bodies = sl.Bodies()
    print(bodies)
    
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
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)
            
            alerts = []
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
            dashboard_data["num_bodies"] = len(bodies.body_list)
            dashboard_data["alerts"] = alerts
            
            viewer.update_view(image, bodies)
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.body_list,
                                body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            
            key = cv2.waitKey(key_wait)
            if key == ord('q'):
                print("Exiting...")
                break
            if key == ord('m'):
                if key_wait > 0:
                    print("Pause")
                    key_wait = 0
                else:
                    print("Restart")
                    key_wait = 10
            if opt.attendance and key == ord('r'):
                report_lines = []
                report_lines.append("----- Lecture Attendance Tracking Report -----")
                report_lines.append("Maximum attendee count: {} students".format(attendance_max))
                report_lines.append("Minimum attendee count: {} students".format(attendance_min))
                for body_id, times in tracked_bodies.items():
                    duration = times['last'] - times['first']
                    report_lines.append("Body ID {}: tracked for {:.2f} seconds".format(body_id, duration))
                report_lines.append("----- End of Report -----")
                dashboard_data["report"] = "\n".join(report_lines)
                print("\n" + dashboard_data["report"] + "\n")
    
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the web dashboard server in a background thread.
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Now run the ZED tracking main loop.
    main(registered_students, covid_bodies)
