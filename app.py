from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import sys
import time
import threading
import numpy as np
import pyzed.sl as sl
import cv_viewer.tracking_viewer as cv_viewer
from predict_hvac import predict_hvac_action
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

print("=== Starting Digital Twin Backend ===")
print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

app = Flask(__name__)
CORS(app)

# Global state to store tracking data
tracking_state = {
    "attendance": "N/A",
    "num_bodies": 0,
    "alerts": [],
    "report": "",
    "hvac": "N/A",
    "tracked_bodies": {},
    "att_max": 0,
    "att_min": float('inf'),
    "registered_students": None,
    "covid_bodies": 12,
    "features": {
        "covid": False,
        "phone": False,
        "attendance": False
    },
    "is_paused": False,
    "is_lecture_active": False,
    "lecture_start_time": None,
    "zed": None,
    "bodies": None,
    "image": None,
    "rt_params": None,
    "bt_params": None,
    "info": None,
    "scale": None
}

tracking_state_lock = threading.Lock()

def initialize_camera(data):
    print("\n=== Camera Initialization Started ===")
    try:
        # Initialize ZED
        tracking_state["zed"] = sl.Camera()
        init = sl.InitParameters()
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init.camera_resolution = sl.RESOLUTION.HD720
        
        print("Opening ZED camera...")
        err = tracking_state["zed"].open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ ERROR: Failed to open ZED camera. Error code: {err}")
            return False
        print("✅ ZED camera opened successfully")

        # Enable tracking
        print("Enabling positional tracking...")
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_area_memory = True
        err = tracking_state["zed"].enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ ERROR: Failed to enable positional tracking. Error code: {err}")
            return False
        print("✅ Positional tracking enabled")

        # Enable body tracking
        print("Configuring body tracking...")
        tracking_state["bt_params"] = sl.BodyTrackingParameters()
        tracking_state["bt_params"].enable_tracking = True
        tracking_state["bt_params"].detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        tracking_state["bt_params"].body_format = sl.BODY_FORMAT.BODY_18
        tracking_state["bt_params"].enable_body_fitting = True

        err = tracking_state["zed"].enable_body_tracking(tracking_state["bt_params"])
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ ERROR: Failed to enable body tracking. Error code: {err}")
            return False
        print("✅ Body tracking enabled")

        # Initialize runtime parameters
        tracking_state["rt_params"] = sl.BodyTrackingRuntimeParameters()
        tracking_state["rt_params"].detection_confidence_threshold = 40

        # Get camera information
        tracking_state["info"] = tracking_state["zed"].get_camera_information()
        tracking_state["display_resolution"] = sl.Resolution(
            min(tracking_state["info"].camera_configuration.resolution.width, 1280),
            min(tracking_state["info"].camera_configuration.resolution.height, 720)
        )

        # Calculate scale for 2D view
        tracking_state["scale"] = [
            tracking_state["display_resolution"].width / tracking_state["info"].camera_configuration.resolution.width,
            tracking_state["display_resolution"].height / tracking_state["info"].camera_configuration.resolution.height
        ]

        # Initialize frame processing variables
        tracking_state["bodies"] = sl.Bodies()
        tracking_state["image"] = sl.Mat()

        print("\n=== Camera Initialization Complete ===")
        return True
    except Exception as e:
        print(f"❌ ERROR during camera initialization: {str(e)}")
        return False

def process_frame():
    print("\n=== Starting Frame Processing Thread ===")
    frame_count = 0
    last_occupancy_update = 0
    last_hvac_update = 0
    occupancy_update_interval = 2.0  # Update occupancy every 2 seconds
    hvac_update_interval = 300.0  # Update HVAC every 5 minutes
    
    while True:
        if tracking_state["is_paused"]:
            time.sleep(0.1)
            continue
            
        try:
            err = tracking_state["zed"].grab()
            if err == sl.ERROR_CODE.SUCCESS:
                # Retrieve image and bodies
                tracking_state["zed"].retrieve_image(tracking_state["image"], sl.VIEW.LEFT, sl.MEM.CPU, tracking_state["display_resolution"])
                tracking_state["zed"].retrieve_bodies(tracking_state["bodies"], tracking_state["rt_params"])
                
                num = len(tracking_state["bodies"].body_list)
                alerts = []
                now = time.time()

                # COVID monitoring (Social Distancing)
                if tracking_state["features"]["covid"]:
                    if num > tracking_state["covid_bodies"]:
                        alerts.append(f"More than {tracking_state['covid_bodies']} bodies detected!")
                    # pairwise distance check
                    positions = [b.position for b in tracking_state["bodies"].body_list]
                    for i in range(len(positions)):
                        for j in range(i+1, len(positions)):
                            if np.linalg.norm(positions[i] - positions[j]) < 1.0:
                                alerts.append("Two bodies < 1m apart!")

                # Phone detection
                if tracking_state["features"]["phone"] and tracking_state["bodies"].body_list:
                    for body in tracking_state["bodies"].body_list:
                        kp = body.keypoint_2d
                        if len(kp) > 1:
                            nose_y = getattr(kp[0], 'y', kp[0][1])
                            neck_y = getattr(kp[1], 'y', kp[1][1])
                            if abs(nose_y - neck_y) < 40:
                                alerts.append(f"Phone usage alert for body {body.id}!")

                # Attendance tracking with debouncing
                if tracking_state["is_lecture_active"] and tracking_state["registered_students"] is not None:
                    present = num
                    tracking_state["att_max"] = max(tracking_state["att_max"], present)
                    tracking_state["att_min"] = min(tracking_state["att_min"], present)
                    
                    # Track individual body durations
                    for b in tracking_state["bodies"].body_list:
                        bid = int(b.id) if hasattr(b, 'id') else hash(b)
                        if bid not in tracking_state["tracked_bodies"]:
                            tracking_state["tracked_bodies"][bid] = {"first": now, "last": now}
                        tracking_state["tracked_bodies"][bid]["last"] = now

                    # Calculate attendance ratio and status
                    ratio = present / tracking_state["registered_students"]
                    if ratio < 1/3:
                        status = "Poor"
                    elif ratio <= 2/3:
                        status = "Fair"
                    else:
                        status = "Good"
                    
                    tracking_state["attendance"] = f"{status} ({present} / {tracking_state['registered_students']})"
                else:
                    tracking_state["attendance"] = "N/A"

                # Update HVAC status with reduced frequency
                if now - last_hvac_update >= hvac_update_interval:
                    try:
                        result = predict_hvac_action(num)
                        tracking_state["hvac"] = result.get('suggestion', 'N/A')
                        last_hvac_update = now
                    except Exception as e:
                        print(f"Error getting HVAC suggestion: {str(e)}")
                        tracking_state["hvac"] = "N/A"

                tracking_state["num_bodies"] = num
                tracking_state["alerts"] = alerts

                frame_count += 1
            else:
                print(f"❌ Failed to grab frame. Error code: {err}")
        except Exception as e:
            print(f"❌ Error processing frame: {str(e)}")
        
        time.sleep(0.1)

# API Routes
@app.route('/api/initialize', methods=['POST'])
def initialize():
    print("\n=== Received Initialization Request ===")
    try:
        data = request.json
        print(f"Received data: {data}")
        
        if not data:
            print("❌ No data received in request")
            return jsonify({"error": "No data received"}), 400
        
        tracking_state["features"]["covid"] = data.get("covid", False)
        tracking_state["features"]["phone"] = data.get("phone", False)
        tracking_state["features"]["attendance"] = data.get("attendance", False)
        
        print(f"Feature flags set:")
        print(f"- COVID monitoring: {tracking_state['features']['covid']}")
        print(f"- Phone detection: {tracking_state['features']['phone']}")
        print(f"- Attendance tracking: {tracking_state['features']['attendance']}")
        
        if tracking_state["features"]["attendance"]:
            registered_students = data.get("registered_students")
            if not registered_students:
                print("❌ Error: Registered students count required when attendance is enabled")
                return jsonify({"error": "registered_students required when attendance is enabled"}), 400
            
            if tracking_state["features"]["covid"] and registered_students > tracking_state["covid_bodies"]:
                print(f"❌ Error: Registered students ({registered_students}) exceeds COVID max ({tracking_state['covid_bodies']})")
                return jsonify({"error": "registered_students exceeds COVID max"}), 400
            if not tracking_state["features"]["covid"] and registered_students > 30:
                print(f"❌ Error: Registered students ({registered_students}) exceeds capacity (30)")
                return jsonify({"error": "registered_students exceeds capacity"}), 400
            
            tracking_state["registered_students"] = registered_students
            print(f"✅ Registered students set to: {registered_students}")

        if not initialize_camera(data):
            print("❌ Camera initialization failed")
            return jsonify({"error": "Failed to initialize camera"}), 500
        
        print("Starting camera processing thread...")
        threading.Thread(target=process_frame, daemon=True).start()
        print("✅ Camera processing thread started")
        return jsonify({"status": "initialized"})
    except Exception as e:
        print(f"❌ Error in initialization endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    status = {
        "num_bodies": tracking_state["num_bodies"],
        "alerts": tracking_state["alerts"],
        "hvac": tracking_state["hvac"],
        "is_paused": tracking_state["is_paused"]
    }
    
    # Only include attendance if it's not N/A
    if tracking_state["attendance"] != "N/A":
        status["attendance"] = tracking_state["attendance"]
        
    return jsonify(status)

@app.route('/api/pause', methods=['POST'])
def pause():
    tracking_state["is_paused"] = True
    return jsonify({"status": "paused"})

@app.route('/api/resume', methods=['POST'])
def resume():
    tracking_state["is_paused"] = False
    return jsonify({"status": "resumed"})

@app.route('/api/start_lecture', methods=['POST'])
def start_lecture():
    if not tracking_state["features"]["attendance"]:
        return jsonify({"error": "Attendance tracking not enabled"}), 400
    
    if tracking_state["is_lecture_active"]:
        return jsonify({"error": "Lecture already in progress"}), 400
    
    tracking_state["is_lecture_active"] = True
    tracking_state["lecture_start_time"] = time.time()
    tracking_state["tracked_bodies"] = {}  # Reset tracking for new lecture
    tracking_state["att_max"] = 0
    tracking_state["att_min"] = float('inf')
    return jsonify({"status": "lecture started"})

@app.route('/api/stop_lecture', methods=['POST'])
def stop_lecture():
    if not tracking_state["is_lecture_active"]:
        return jsonify({"error": "No active lecture"}), 400
    
    tracking_state["is_lecture_active"] = False
    
    # Generate final report
    lines = [
        "----- Lecture Attendance Tracking Report -----",
        f"Lecture Duration: {time.time() - tracking_state['lecture_start_time']:.2f} seconds",
        f"Max attendees: {tracking_state['att_max']}",
        f"Min attendees: {tracking_state['att_min']}",
        f"Registered Students: {tracking_state['registered_students']}",
        "\nIndividual Tracking:"
    ]
    
    for bid, times in tracking_state["tracked_bodies"].items():
        dur = times["last"] - times["first"]
        lines.append(f"Body {bid}: {dur:.2f}s")
    
    lines.append("----- End of Report -----")
    tracking_state["report"] = '\n'.join(lines)
    
    return jsonify({
        "status": "lecture stopped",
        "report": tracking_state["report"]
    })

@app.route('/api/hvac', methods=['GET'])
def get_hvac():
    try:
        result = predict_hvac_action(tracking_state["num_bodies"])
        tracking_state["hvac"] = result.get('suggestion', 'No suggestion')
        return jsonify({"hvac": tracking_state["hvac"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update_features', methods=['POST'])
def update_features():
    try:
        data = request.get_json()
        print(f"Received feature update request: {data}")  # Debug logging
        
        # Get values with explicit type checking, supporting both parameter names
        phone_detection = data.get('phone_detection', data.get('phone'))
        if phone_detection is None:
            print("Warning: phone_detection/phone not provided in request")
            return jsonify({'error': 'phone_detection/phone value is required'}), 400
            
        social_distancing = data.get('social_distancing', data.get('covid', False))
        attendance = data.get('attendance', False)
        
        # Handle registered_students more robustly
        registered_students = data.get('registered_students', 0)
        try:
            registered_students = int(registered_students) if registered_students else 0
        except (ValueError, TypeError):
            registered_students = 0
        
        # Update tracking state with thread safety
        with tracking_state_lock:
            # Only update the features that were sent
            if 'phone' in data or 'phone_detection' in data:
                tracking_state['features']['phone'] = bool(phone_detection)
            if 'covid' in data or 'social_distancing' in data:
                tracking_state['features']['covid'] = bool(social_distancing)
            if 'attendance' in data:
                tracking_state['features']['attendance'] = bool(attendance)
            if 'registered_students' in data:
                tracking_state['registered_students'] = registered_students
            
        # Log only the relevant state changes
        print(f"Updated features: {tracking_state['features']}")
        print(f"Updated registered_students: {tracking_state['registered_students']}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error updating features: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 