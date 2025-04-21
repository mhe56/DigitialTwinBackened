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

import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Set up initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open: " + repr(err) + ". Exit program.")
        exit()

    # Set up body tracking parameters
    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    body_params.enable_body_fitting = True

    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True  # Uncomment if needed
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Body tracking: Loading Module...")
    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable Body Tracking: " + repr(err) + ". Exit program.")
        zed.close()
        exit()

    bodies = sl.Bodies()
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    # Adjust detection confidence threshold as needed (e.g., indoor: ~50+, outdoor: ~20-30)
    body_runtime_param.detection_confidence_threshold = 40

    # Infinite loop to continuously capture frames
    frame_counter = 0
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            frame_counter += 1
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            if bodies.is_new:
                body_array = bodies.body_list
                if len(body_array) > 0:
                    first_body = body_array[0]
                    # Print some basic attributes for reference
                    # print(f"Frame {frame_counter}: {len(body_array)} person(s) detected.")
                    # print("  Confidence (" + str(int(first_body.confidence)) + "/100)")
                    # if body_params.enable_tracking:
                    #     print("  Tracking ID: " + str(int(first_body.id)) +
                    #           ", tracking state: " + repr(first_body.tracking_state) +
                    #           ", action state: " + repr(first_body.action_state))
                    
                    # Check keypoints
                    keypoint_2d = first_body.keypoint_2d
                    if len(keypoint_2d) > 1:
                        try:
                            # Assuming keypoint_2d is a list of coordinates: [x, y]
                            nose_y = keypoint_2d[0][1]
                            neck_y = keypoint_2d[1][1]
                        except Exception as e:
                            try:
                                # Fallback in case of attribute access
                                nose_y = keypoint_2d[0].y
                                neck_y = keypoint_2d[1].y
                            except Exception as ex:
                                nose_y = None
                                neck_y = None
                        # If successfully retrieved, check if the keypoints are close
                        if nose_y is not None and neck_y is not None:
                            threshold = 50  # Adjust this threshold based on your use case
                            if abs(nose_y - neck_y) < threshold:
                                print(f"Alert at Frame {frame_counter}: Nose and Neck keypoints are close - person might be using their phone in class!")
                                
    # These lines will never be reached as the loop runs infinitely,
    # but they are placed here for completeness.
    zed.disable_body_tracking()
    zed.close()

if __name__ == "__main__":
    main()
