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
       must not exceed 12 (the maximum allowed bodies for the COVID-19 scenario). Otherwise,
       when the COVID-19 scenario is not enabled, the registered student count must be <= 30.
"""

import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse

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
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Set up initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # Set up body tracking parameters
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across image flow
    body_param.enable_body_fitting = False           # Smooth skeleton movement
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18     # Choose the BODY_FORMAT you wish to use

    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                       min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width,
                   display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam,
                body_param.enable_tracking, body_param.body_format)
    
    # Create bodies object to store detection results
    bodies = sl.Bodies()
    print(bodies)

    image = sl.Mat()
    key_wait = 10 
    frame_counter = 0

    while viewer.is_available():
        frame_counter += 1

        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image and body tracking results
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)
            
            # ---------------- Alert Logic for COVID-19 Scenario ----------------
            if opt.covid:
                num_bodies = len(bodies.body_list)
                if num_bodies > covid_bodies:
                    print("Alert at frame {}: More than 12 bodies detected in the room!".format(frame_counter))
                for i in range(num_bodies):
                    for j in range(i + 1, num_bodies):
                        pos1 = bodies.body_list[i].position
                        pos2 = bodies.body_list[j].position
                        if np.linalg.norm(pos1 - pos2) < 1.0:
                            print("Alert at frame {}: Two bodies are less than 1 meter apart!".format(frame_counter))
            # --------------------------------------------------------------------

            # ---------------- Alert Logic for Phone Detection Scenario ----------------
            if opt.phone:
                if bodies.is_new:
                    body_array = bodies.body_list
                    if len(body_array) > 0:
                        first_body = body_array[0]
                        keypoint_2d = first_body.keypoint_2d
                        if len(keypoint_2d) > 1:
                            try:
                                # Attempt list-based access (e.g., [x, y])
                                nose_y = keypoint_2d[0][1]
                                neck_y = keypoint_2d[1][1]
                            except Exception:
                                try:
                                    # Fallback to attribute-based access
                                    nose_y = keypoint_2d[0].y
                                    neck_y = keypoint_2d[1].y
                                except Exception:
                                    nose_y = None
                                    neck_y = None
                            if nose_y is not None and neck_y is not None:
                                threshold = 40  # Adjust threshold as needed
                                if abs(nose_y - neck_y) < threshold:
                                    print(f"Alert at Frame {frame_counter}: Nose and Neck keypoints are close - person might be using their phone in class!")
            # -------------------------------------------------------------------------

            # ---------------- Attendance Classification Scenario ----------------
            if opt.attendance and registered_students is not None:
                present_count = len(bodies.body_list)
                ratio = present_count / registered_students
                if ratio < (1/3):
                    classification = "Poor"
                elif ratio <= (2/3):
                    classification = "Fair"
                else:
                    classification = "Good"
                print("Attendance at frame {}: {}/{} students present. Classification: {}"
                      .format(frame_counter, present_count, registered_students, classification))
            # -------------------------------------------------------------------------

            # Update OpenGL view and 2D image viewer
            viewer.update_view(image, bodies) 
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.body_list,
                                body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            key = cv2.waitKey(key_wait)
            if key == 113:  # for 'q' key
                print("Exiting...")
                break
            if key == 109:  # for 'm' key (pause/restart)
                if key_wait > 0:
                    print("Pause")
                    key_wait = 0 
                else:
                    print("Restart")
                    key_wait = 10 

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
    # New arguments for enabling scenarios
    parser.add_argument('-c', '--covid', action='store_true', help='Enable COVID-19 scenario alerts')
    parser.add_argument('-p', '--phone', action='store_true', help='Enable phone detection scenario')
    parser.add_argument('-a', '--attendance', action='store_true', help='Enable attendance classification scenario')
    opt = parser.parse_args()
    
    # If attendance classification is enabled, ask for the number of registered students.
    registered_students = None
    covid_bodies=25

    if opt.attendance:
        try:
            registered_students = int(input("Enter number of registered students: "))
        except ValueError:
            print("Invalid input. Please enter an integer value for registered students.")
            exit(1)
        if opt.covid:
            # For the COVID-19 scenario, allow a maximum of 12 registered students.
            if registered_students > covid_bodies:
                print("Error: the covid-19 scenario is enabled. Maximum number of registered students exceeded!")
                exit(1)
        else:
            # Without the COVID-19 scenario, the lab capacity is 30.
            if registered_students > 30:
                print("Error: Lab capacity exceeded! Maximum number of registered students is 30.")
                exit(1)
    
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use a wired camera. Exiting program.")
        exit()
        
    main(registered_students, covid_bodies)
