#!/usr/bin/env python
"""
Combined Body Tracking and Custom Skeleton Visualization

This script uses the ZED camera to track human bodies, converts the
COCO-18 keypoints to a custom 17-joint skeleton, and visualizes the new
skeleton on a blank window (without the camera frame).
"""

import cv2
import sys
import argparse
import numpy as np
import pyzed.sl as sl
import time

# ------------------ Conversion Function ------------------ #
def convert_body18_to_custom17(coco18_keypoints):
    """
    Convert a single frame of COCO-18 keypoints (shape: (18,3))
    to a custom 17-joint skeleton (shape: (17,3)), indexed as:
      0 : Head-top        (approx midpoint of ears)
      1 : Neck            (COCO #1)
      2 : Left shoulder   (COCO #5)
      3 : Left elbow      (COCO #6)
      4 : Left wrist      (COCO #7)
      5 : Right shoulder  (COCO #2)
      6 : Right elbow     (COCO #3)
      7 : Right wrist     (COCO #4)
      8 : Left hip        (COCO #11)
      9 : Left knee       (COCO #12)
     10 : Left ankle      (COCO #13)
     11 : Right hip       (COCO #8)
     12 : Right knee      (COCO #9)
     13 : Right ankle     (COCO #10)
     14 : Pelvis          (midpoint of left & right hip)
     15 : Spine           (midpoint of neck and pelvis)
     16 : Nose            (as an extra head point/fallback)
    """
    custom_keypoints = np.full((17, 3), np.nan, dtype=np.float32)

    # COCO indices
    NOSE       = 0
    NECK       = 1
    R_SHOULDER = 2
    R_ELBOW    = 3
    R_WRIST    = 4
    L_SHOULDER = 5
    L_ELBOW    = 6
    L_WRIST    = 7
    R_HIP      = 8
    R_KNEE     = 9
    R_ANKLE    = 10
    L_HIP      = 11
    L_KNEE     = 12
    L_ANKLE    = 13
    R_EYE      = 14
    L_EYE      = 15
    R_EAR      = 16
    L_EAR      = 17

    def midpoint(pt1, pt2):
        if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
            return np.full((3,), np.nan, dtype=np.float32)
        return 0.5 * (pt1 + pt2)

    # 0: Head-top => midpoint of the ears (or fallback)
    left_ear  = coco18_keypoints[L_EAR]
    right_ear = coco18_keypoints[R_EAR]
    ears_mid  = midpoint(left_ear, right_ear)
    if np.any(np.isnan(ears_mid)):
        eyes_mid = midpoint(coco18_keypoints[L_EYE], coco18_keypoints[R_EYE])
        if np.any(np.isnan(eyes_mid)):
            ears_mid = coco18_keypoints[NOSE]  # final fallback
        else:
            ears_mid = eyes_mid
    custom_keypoints[0] = ears_mid

    # 1: Neck
    custom_keypoints[1] = coco18_keypoints[NECK]
    # 2: Left shoulder
    custom_keypoints[2] = coco18_keypoints[L_SHOULDER]
    # 3: Left elbow
    custom_keypoints[3] = coco18_keypoints[L_ELBOW]
    # 4: Left wrist
    custom_keypoints[4] = coco18_keypoints[L_WRIST]
    # 5: Right shoulder
    custom_keypoints[5] = coco18_keypoints[R_SHOULDER]
    # 6: Right elbow
    custom_keypoints[6] = coco18_keypoints[R_ELBOW]
    # 7: Right wrist
    custom_keypoints[7] = coco18_keypoints[R_WRIST]
    # 8: Left hip
    custom_keypoints[8] = coco18_keypoints[L_HIP]
    # 9: Left knee
    custom_keypoints[9] = coco18_keypoints[L_KNEE]
    # 10: Left ankle
    custom_keypoints[10] = coco18_keypoints[L_ANKLE]
    # 11: Right hip
    custom_keypoints[11] = coco18_keypoints[R_HIP]
    # 12: Right knee
    custom_keypoints[12] = coco18_keypoints[R_KNEE]
    # 13: Right ankle
    custom_keypoints[13] = coco18_keypoints[R_ANKLE]
    # 14: Pelvis (midpoint of left and right hip)
    pelvis = midpoint(coco18_keypoints[L_HIP], coco18_keypoints[R_HIP])
    custom_keypoints[14] = pelvis
    # 15: Spine (midpoint of neck and pelvis)
    spine  = midpoint(coco18_keypoints[NECK], pelvis)
    custom_keypoints[15] = spine
    # 16: Nose (as extra head point)
    custom_keypoints[16] = coco18_keypoints[NOSE]

    return custom_keypoints

# ------------------ Define Skeleton Connectivity ------------------ #
skeleton_connections = [
    (0, 1),   # Head-top to Neck
    (1, 2),   # Neck to Left shoulder
    (2, 3),   # Left shoulder to Left elbow
    (3, 4),   # Left elbow to Left wrist
    (1, 5),   # Neck to Right shoulder
    (5, 6),   # Right shoulder to Right elbow
    (6, 7),   # Right elbow to Right wrist
    (1, 15),  # Neck to Spine
    (15, 14), # Spine to Pelvis
    (14, 8),  # Pelvis to Left hip
    (8, 9),   # Left hip to Left knee
    (9, 10),  # Left knee to Left ankle
    (14, 11), # Pelvis to Right hip
    (11, 12), # Right hip to Right knee
    (12, 13), # Right knee to Right ankle
    (0, 16)   # Optionally, Head-top to Nose (extra head point)
]

# ------------------ Parse Command-Line Arguments ------------------ #
def parse_args(init_params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='',
                        help='Path to an .svo file, if you want to replay it')
    parser.add_argument('--ip_address', type=str, default='',
                        help='IP Address, in format a.b.c.d:port or a.b.c.d, for streaming')
    parser.add_argument('--resolution', type=str, default='HD1080',
                        help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, or VGA')
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init_params.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input:", opt.input_svo_file)
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            ip, port = ip_str.split(':')
            init_params.set_from_stream(ip, int(port))
            print("[Sample] Using Stream input, IP:", ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init_params.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP:", ip_str)
        else:
            print("Invalid IP format. Using live camera stream.")

    res_str = opt.resolution.upper()
    if "HD2K" in res_str:
        init_params.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using resolution HD2K")
    elif "HD1200" in res_str:
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using resolution HD1200")
    elif "HD1080" in res_str:
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using resolution HD1080")
    elif "HD720" in res_str:
        init_params.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using resolution HD720")
    elif "SVGA" in res_str:
        init_params.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using resolution SVGA")
    elif "VGA" in res_str:
        init_params.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using resolution VGA")
    else:
        print("[Sample] No valid resolution entered. Using default.")

    return opt

# ------------------ Main Function ------------------ #
def main():
    print("Running Body Tracking with Custom Skeleton visualization...")
    print("Press 'q' to quit or 'm' to pause/restart the display.")

    # Create a ZED Camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Parse command-line arguments (if any)
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open Error:", err)
        exit(1)

    # Enable positional tracking
    pos_tracking_params = sl.PositionalTrackingParameters()
    # Uncomment the following line if the camera is static for improved performance:
    # pos_tracking_params.set_as_static = True
    zed.enable_positional_tracking(pos_tracking_params)

    # Set up body tracking parameters
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True
    body_param.enable_body_fitting = False
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Expecting COCO-18 format

    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get camera information to set display resolution for our blank canvas
    camera_info = zed.get_camera_information()
    display_width  = min(camera_info.camera_configuration.resolution.width, 1280)
    display_height = min(camera_info.camera_configuration.resolution.height, 720)

    key_wait = 10  # Delay for cv2.waitKey (in ms)
    paused = False

    cv2.namedWindow("Converted Skeleton", cv2.WINDOW_AUTOSIZE)

    # Main loop
    while True:
        if not paused:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                bodies = sl.Bodies()
                zed.retrieve_bodies(bodies, body_runtime_param)

                # Create a blank canvas (black image)
                canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

                for body in bodies.body_list:
                    # Use keypoint_2d which is a numpy array list
                    if len(body.keypoint_2d) < 18:
                        continue
                    coco18 = np.zeros((18, 3), dtype=np.float32)
                    for i in range(18):
                        kp = body.keypoint_2d[i]
                        if len(kp) < 3:
                            # If only x and y are provided, use a default confidence (e.g., 1.0)
                            coco18[i] = [kp[0], kp[1], 1.0]
                        else:
                            coco18[i] = [kp[0], kp[1], kp[2]]

                    custom17 = convert_body18_to_custom17(coco18)

                    # Draw keypoints as circles and label them
                    for idx, pt in enumerate(custom17):
                        if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < display_width and 0 <= y < display_height:
                                cv2.circle(canvas, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                                cv2.putText(canvas, f"{idx}", (x+5, y-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # Draw skeleton connections between joints
                    for (i, j) in skeleton_connections:
                        if i < len(custom17) and j < len(custom17):
                            pt1 = custom17[i]
                            pt2 = custom17[j]
                            if (not np.isnan(pt1[0]) and not np.isnan(pt1[1]) and
                                not np.isnan(pt2[0]) and not np.isnan(pt2[1])):
                                p1 = (int(pt1[0]), int(pt1[1]))
                                p2 = (int(pt2[0]), int(pt2[1]))
                                cv2.line(canvas, p1, p2, color=(255, 0, 0), thickness=2)

                cv2.imshow("Converted Skeleton", canvas)
        key = cv2.waitKey(key_wait) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        if key == ord('m'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    # Clean up resources
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
