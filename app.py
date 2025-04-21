from flask import Flask, request, jsonify
import cv2
import sys
import time
import threading
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl
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

app = Flask(__name__)

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
    "viewer": None,
    "display_resolution": None
}

# Initialize ZED camera
zed = sl.Camera()
init = sl.InitParameters()
init.coordinate_units = sl.UNIT.METER
init.depth_mode = sl.DEPTH_MODE.ULTRA
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

def initialize_camera():
    logging.info("Attempting to initialize ZED camera...")
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        logging.error("Failed to open ZED camera")
        return False
    
    logging.info("ZED camera opened successfully")
    
    # Enable tracking
    logging.info("Enabling positional tracking...")
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    
    logging.info("Configuring body tracking parameters...")
    bt_params = sl.BodyTrackingParameters()
    bt_params.enable_tracking = True
    bt_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    bt_params.body_format = sl.BODY_FORMAT.BODY_18
    
    logging.info("Enabling body tracking...")
    zed.enable_body_tracking(bt_params)
    
    # Initialize viewer
    info = zed.get_camera_information()
    tracking_state["display_resolution"] = sl.Resolution(
        min(info.camera_configuration.resolution.width, 1280),
        min(info.camera_configuration.resolution.height, 720)
    )
    
    logging.info("Initializing GL viewer...")
    tracking_state["viewer"] = gl.GLViewer()
    tracking_state["viewer"].init(
        info.camera_configuration.calibration_parameters.left_cam,
        bt_params.enable_tracking,
        bt_params.body_format
    )
    
    logging.info("Camera initialization complete")
    return True

def process_frame():
    logging.info("Starting frame processing thread")
    bodies = sl.Bodies()
    image = sl.Mat()
    rt_params = sl.BodyTrackingRuntimeParameters()
    rt_params.detection_confidence_threshold = 40

    while True:
        if tracking_state["is_paused"]:
            time.sleep(0.1)
            continue
            
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, tracking_state["display_resolution"])
            zed.retrieve_bodies(bodies, rt_params)
            num = len(bodies.body_list)
            alerts = []

            # COVID monitoring
            if tracking_state["features"]["covid"]:
                if num > tracking_state["covid_bodies"]:
                    alert = f"More than {tracking_state['covid_bodies']} bodies detected!"
                    alerts.append(alert)
                    logging.warning(alert)
                positions = [b.position for b in bodies.body_list]
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        if np.linalg.norm(positions[i] - positions[j]) < 1.0:
                            alert = "Two bodies < 1m apart!"
                            alerts.append(alert)
                            logging.warning(alert)

            # Phone detection
            if tracking_state["features"]["phone"] and bodies.is_new and bodies.body_list:
                kp = bodies.body_list[0].keypoint_2d
                if len(kp) > 1:
                    nose_y = getattr(kp[0], 'y', kp[0][1])
                    neck_y = getattr(kp[1], 'y', kp[1][1])
                    if abs(nose_y - neck_y) < 40:
                        alert = "Phone usage alert!"
                        alerts.append(alert)
                        logging.warning(alert)

            # Attendance tracking
            if tracking_state["features"]["attendance"] and tracking_state["registered_students"] is not None:
                present = num
                tracking_state["att_max"] = max(tracking_state["att_max"], present)
                tracking_state["att_min"] = min(tracking_state["att_min"], present)
                now = time.time()
                
                for b in bodies.body_list:
                    bid = int(b.id) if hasattr(b, 'id') else hash(b)
                    if bid not in tracking_state["tracked_bodies"]:
                        tracking_state["tracked_bodies"][bid] = {"first": now, "last": now}
                    tracking_state["tracked_bodies"][bid]["last"] = now

                ratio = tracking_state["att_max"] / tracking_state["registered_students"]
                cls = "Poor" if ratio < 1/3 else "Fair" if ratio <= 2/3 else "Good"
                tracking_state["attendance"] = f"{cls} ({present} / {tracking_state['registered_students']})"
                logging.info(f"Attendance update: {tracking_state['attendance']}")

            tracking_state["num_bodies"] = num
            tracking_state["alerts"] = alerts

            # Update viewers
            if tracking_state["viewer"] and tracking_state["viewer"].is_available():
                tracking_state["viewer"].update_view(image, bodies)
                
            # Update 2D view
            frame = image.get_data()
            cv_viewer.render_2D(frame, 
                              [tracking_state["display_resolution"].width / info.camera_configuration.resolution.width,
                               tracking_state["display_resolution"].height / info.camera_configuration.resolution.height],
                              bodies.body_list,
                              bt_params.enable_tracking,
                              bt_params.body_format)
            cv2.imshow("ZED | 2D View", frame)
            cv2.waitKey(1)

            time.sleep(0.1)  # Prevent CPU overuse

# API Routes
@app.route('/api/initialize', methods=['POST'])
def initialize():
    logging.info("Received initialization request")
    data = request.json
    tracking_state["features"]["covid"] = data.get("covid", False)
    tracking_state["features"]["phone"] = data.get("phone", False)
    tracking_state["features"]["attendance"] = data.get("attendance", False)
    
    logging.info(f"Feature flags set: COVID={tracking_state['features']['covid']}, "
                f"Phone={tracking_state['features']['phone']}, "
                f"Attendance={tracking_state['features']['attendance']}")
    
    if tracking_state["features"]["attendance"]:
        registered_students = data.get("registered_students")
        if not registered_students:
            logging.error("Registered students count required when attendance is enabled")
            return jsonify({"error": "registered_students required when attendance is enabled"}), 400
        
        if tracking_state["features"]["covid"] and registered_students > tracking_state["covid_bodies"]:
            logging.error(f"Registered students ({registered_students}) exceeds COVID max ({tracking_state['covid_bodies']})")
            return jsonify({"error": "registered_students exceeds COVID max"}), 400
        if not tracking_state["features"]["covid"] and registered_students > 30:
            logging.error(f"Registered students ({registered_students}) exceeds capacity (30)")
            return jsonify({"error": "registered_students exceeds capacity"}), 400
        
        tracking_state["registered_students"] = registered_students
        logging.info(f"Registered students set to: {registered_students}")

    if not initialize_camera():
        return jsonify({"error": "Failed to initialize camera"}), 500
    
    # Start processing thread
    threading.Thread(target=process_frame, daemon=True).start()
    logging.info("Camera processing thread started")
    return jsonify({"status": "initialized"})

@app.route('/api/status', methods=['GET'])
def get_status():
    logging.info("Status request received")
    return jsonify({
        "attendance": tracking_state["attendance"],
        "num_bodies": tracking_state["num_bodies"],
        "alerts": tracking_state["alerts"],
        "hvac": tracking_state["hvac"],
        "is_paused": tracking_state["is_paused"]
    })

@app.route('/api/pause', methods=['POST'])
def pause():
    logging.info("Pause request received")
    tracking_state["is_paused"] = True
    return jsonify({"status": "paused"})

@app.route('/api/resume', methods=['POST'])
def resume():
    logging.info("Resume request received")
    tracking_state["is_paused"] = False
    return jsonify({"status": "resumed"})

@app.route('/api/report', methods=['GET'])
def get_report():
    logging.info("Report request received")
    if not tracking_state["features"]["attendance"]:
        logging.error("Attendance tracking not enabled")
        return jsonify({"error": "Attendance tracking not enabled"}), 400
    
    lines = [
        "----- Lecture Attendance Tracking Report -----",
        f"Max attendees: {tracking_state['att_max']}",
        f"Min attendees: {tracking_state['att_min']}"
    ]
    
    for bid, times in tracking_state["tracked_bodies"].items():
        dur = times["last"] - times["first"]
        lines.append(f"Body {bid}: {dur:.2f}s")
    
    lines.append("----- End of Report -----")
    tracking_state["report"] = '\n'.join(lines)
    logging.info("Report generated successfully")
    return jsonify({"report": tracking_state["report"]})

@app.route('/api/hvac', methods=['GET'])
def get_hvac():
    logging.info("HVAC request received")
    try:
        result = predict_hvac_action(tracking_state["num_bodies"])
        tracking_state["hvac"] = result.get('suggestion', 'No suggestion')
        logging.info(f"HVAC suggestion: {tracking_state['hvac']}")
        return jsonify({"hvac": tracking_state["hvac"]})
    except Exception as e:
        logging.error(f"HVAC prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask application")
    app.run(debug=True, port=5000) 