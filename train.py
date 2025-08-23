import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import logging
import json
import time
from collections import deque
import threading
import urllib.parse
from flask import Flask, Response, render_template, jsonify, request
import glob


class HelmetDetectionSystem:
    def __init__(self, config_path="config.json"):
        """Initialize the helmet detection system with configuration"""
        self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        self.load_model()
        self.active_violations = {}
        self.frame_buffer = deque(maxlen=30)
        self.last_screenshot_time = {}
        self.violation_id_counter = 0
        self.frame_skip = 2
        self.frame_count = 0
        self.last_frame = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.detection_active = False
        self.total_violations = 0
        self.current_fps = 0
        self.last_violation_path = None
        self.detection_thread = None

    def load_config(self, config_path):
        """Load configuration from JSON file with multiple camera support"""
        default_config = {
        "model_path": "weights/best.pt",
            "cameras": {
                "camera_1": {
                    "name": "Main Entrance",
                    "url": "rtsp://admin:admin%40123@103.69.44.194:554/cam/realmonitor?channel=1&subtype=0"
                },
                "camera_2": {
                    "name": "ISEJ",
                    "url": "rtsp://admin:admin%40123@103.69.44.194:554/cam/realmonitor?channel=2&subtype=0"
                },
                "camera_3": {
                    "name": "Metal Liners",
                    "url": "rtsp://admin:admin%40123@103.69.44.194:554/cam/realmonitor?channel=3&subtype=0"
                },
                "camera_4": {
                    "name": "Guard Room",
                    "url": "rtsp://admin:admin%40123@103.69.44.194:554/cam/realmonitor?channel=4&subtype=0"
                }
            },
            "active_camera": "camera_1",
            "use_opencv_fallback": True,
            "save_dir": "Violations",
            "log_dir": "Logs",
            "frame_width": 1920,
            "frame_height": 1080,
            "display_width": 1280,
            "display_height": 720,
            "confidence_threshold": 0.5,
            "movement_threshold": 100,
            "stabilization_time": 2.0,
            "violation_cooldown_seconds": 30,
            "max_reconnect_attempts": 5,
            "rtsp_timeout": 10
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Error loading config file: {e}. Using default configuration.")
        else:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config file: {config_path}")

        self.config = default_config

        # Validate configuration
        if not os.path.exists(self.config["model_path"]):
            raise FileNotFoundError(f"Model file not found: {self.config['model_path']}")

        if self.config["active_camera"] not in self.config["cameras"]:
            print(f"Warning: Active camera '{self.config['active_camera']}' not found!")
            self.config["active_camera"] = list(self.config["cameras"].keys())[0]
            print(f"Switched to default camera: {self.config['active_camera']}")

    def setup_logging(self):
        """Setup logging system with UTF-8 encoding"""
        log_dir = self.config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{log_dir}/helmet_detection_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Helmet Detection System Started")
        camera_info = self.get_current_camera_info()
        self.logger.info(f"Active Camera: {camera_info['id']} - {camera_info['name']} - {camera_info['url']}")

    def setup_directories(self):
        """Create necessary directories"""
        base_save_dir = self.config["save_dir"]
        os.makedirs(base_save_dir, exist_ok=True)
        os.makedirs(self.config["log_dir"], exist_ok=True)
        for cam_id, cam_info in self.config["cameras"].items():
            cam_dir = f"{base_save_dir}/{cam_id}_{cam_info['name'].replace(' ', '_')}"
            os.makedirs(cam_dir, exist_ok=True)
            os.makedirs(f"{cam_dir}/daily", exist_ok=True)

    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.config["model_path"])
            self.logger.info(f"Model loaded successfully. Classes: {self.model.names}")
            print(f"Model loaded successfully. Classes: {self.model.names}")
            self.detection_active = True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            print(f"Failed to load model: {e}")
            raise

    def get_current_camera_info(self):
        """Get current active camera information"""
        active_cam = self.config["active_camera"]
        return {
            "id": active_cam,
            "name": self.config["cameras"][active_cam]["name"],
            "url": self.config["cameras"][active_cam]["url"]
        }

    def get_cameras_list(self):
        """Get list of all cameras with active status"""
        cameras = []
        for cam_id, cam_info in self.config["cameras"].items():
            cameras.append({
                "id": cam_id,
                "name": cam_info["name"],
                "url": cam_info["url"],
                "active": cam_id == self.config["active_camera"]
            })
        return cameras

    def switch_camera(self, camera_id):
        """Switch to a different camera"""
        if camera_id not in self.config["cameras"]:
            return False, f"Camera {camera_id} not found"
        
        was_running = self.is_running
        if was_running:
            self.stop()
            time.sleep(1)
        
        self.config["active_camera"] = camera_id
        camera_info = self.get_current_camera_info()
        self.logger.info(f"Switched to camera: {camera_info['name']}")
        
        if was_running:
            self.start_detection_thread()
        
        return True, f"Switched to {camera_info['name']}"

    def create_opencv_capture(self):
        """Create OpenCV VideoCapture"""
        try:
            camera_info = self.get_current_camera_info()
            url = camera_info["url"]
            try:
                url = int(url)
            except ValueError:
                pass

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise Exception(f"Could not open camera: {url}")

            if isinstance(url, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.logger.info(f"OpenCV capture started for {camera_info['name']}")
            print(f"OpenCV capture started for {camera_info['name']}")
            return cap
        except Exception as e:
            self.logger.error(f"Failed to create OpenCV capture: {e}")
            print(f"Failed to create OpenCV capture: {e}")
            raise

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def calculate_box_distance(self, box1, box2):
        """Calculate distance between two bounding boxes"""
        cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
        cx2, cy2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
        return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

    def assign_violation_ids(self, current_violations):
        """Assign unique IDs to violations based on position tracking"""
        current_time = time.time()
        assigned_violations = {}
        expired_ids = [vid for vid, data in self.active_violations.items()
                       if current_time - data['last_seen'] > 5.0]
        for vid in expired_ids:
            del self.active_violations[vid]

        for violation_box in current_violations:
            best_match_id = None
            best_iou = 0.0
            best_distance = float('inf')
            for existing_id, existing_data in self.active_violations.items():
                existing_box = existing_data['box']
                iou = self.calculate_iou(violation_box, existing_box)
                distance = self.calculate_box_distance(violation_box, existing_box)
                if iou > 0.3 and iou > best_iou:
                    best_match_id = existing_id
                    best_iou = iou
                elif iou <= 0.3 and distance < 150 and distance < best_distance:
                    best_match_id = existing_id
                    best_distance = distance

            if best_match_id:
                self.active_violations[best_match_id].update({
                    'box': violation_box,
                    'last_seen': current_time
                })
                assigned_violations[best_match_id] = violation_box
            else:
                self.violation_id_counter += 1
                new_id = self.violation_id_counter
                self.active_violations[new_id] = {
                    'box': violation_box,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'screenshot_taken': False
                }
                assigned_violations[new_id] = violation_box
        return assigned_violations

    def process_detections(self, frame, results):
        """Process YOLO detection results"""
        annotated = frame.copy()
        violations = []
        detection_count = 0
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                detection_count += len(result.boxes)
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    if conf < self.config["confidence_threshold"]:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    cls = int(cls)
                    label = self.model.names[cls]
                    if cls == 1:  # NO-Hardhat
                        color = (0, 0, 255)
                        violations.append((x1, y1, x2, y2))
                        cv2.putText(annotated, "NO HELMET", (x1, y1 - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:  # Hardhat
                        color = (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        with self.frame_lock:
            self.last_frame = annotated.copy()
        return annotated, violations

    def should_save_violation(self, assigned_violations):
        """Decide when to save violation screenshot with movement + cooldown logic"""
        current_time = time.time()
        violations_to_save = []

        for violation_id, violation_box in assigned_violations.items():
            violation_data = self.active_violations[violation_id]

            # First time → save immediately
            if not violation_data.get('screenshot_taken'):
                violation_data['screenshot_taken'] = True
                violation_data['last_saved'] = current_time
                violations_to_save.append((violation_id, violation_box))
                continue

            # Movement check (agar 50px se zyada shift hua to turant save)
            prev_box = violation_data['box']
            movement = self.calculate_box_distance(violation_box, prev_box)

            if movement > 50:
                violation_data['last_saved'] = current_time
                violations_to_save.append((violation_id, violation_box))
                continue

            # Static banda → har 5 min (300 sec) me ek screenshot
            if current_time - violation_data.get('last_saved', 0) > 300:
                violation_data['last_saved'] = current_time
                violations_to_save.append((violation_id, violation_box))

        return violations_to_save

    def save_violation_screenshot(self, frame, violation_box):
        """Save violation screenshot for NO-Hardhat"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        date_folder = datetime.now().strftime("%Y%m%d")
        camera_info = self.get_current_camera_info()
        cam_dir = f"{self.config['save_dir']}/{camera_info['id']}_{camera_info['name'].replace(' ', '_')}"
        daily_dir = f"{cam_dir}/daily/{date_folder}"
        os.makedirs(daily_dir, exist_ok=True)
        filename = f"{daily_dir}/NO_HELMET_{camera_info['id']}_{timestamp}.jpg"
        cv2.putText(frame, f"NO HELMET VIOLATION - {camera_info['name']}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Camera: {camera_info['id']} - {camera_info['name']}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x1, y1, x2, y2 = violation_box
        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
        cv2.putText(frame, "NO HELMET", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"NO HELMET violation saved: {filename}")
            print(f"NO HELMET violation saved: {filename}")
            self.last_violation_path = filename
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
            return None

    def get_violation_info(self):
        """Get violation directory and file information"""
        camera_info = self.get_current_camera_info()
        cam_dir = f"{self.config['save_dir']}/{camera_info['id']}_{camera_info['name'].replace(' ', '_')}"
        
        info = {
            "save_directory": os.path.abspath(cam_dir),
            "directory_exists": os.path.exists(cam_dir),
            "total_files": 0,
            "last_violation": self.last_violation_path
        }
        
        if os.path.exists(cam_dir):
            pattern = f"{cam_dir}/daily/*/*.jpg"
            files = glob.glob(pattern)
            info["total_files"] = len(files)
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                info["last_violation"] = files[0] if not self.last_violation_path else self.last_violation_path
        
        return info

    def get_stats(self):
        """Get current system statistics"""
        camera_info = self.get_current_camera_info()
        return {
            "status": "running" if self.is_running else "stopped",
            "total_violations": self.total_violations,
            "active_violations": len(self.active_violations),
            "fps": self.current_fps,
            "camera_name": camera_info["name"],
            "camera_id": camera_info["id"],
            "last_violation_path": self.last_violation_path
        }

    def reset_violations(self):
        """Reset violation counters"""
        self.total_violations = 0
        self.active_violations.clear()
        self.last_screenshot_time.clear()
        self.violation_id_counter = 0
        self.last_violation_path = None
        self.logger.info("Violation counters reset")

    def run_detection(self):
        """Main detection loop"""
        capture_source = None
        reconnect_attempts = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        print("Starting helmet detection...")
        self.is_running = True
        try:
            capture_source = self.create_opencv_capture()
            while self.is_running:
                try:
                    ret, frame = capture_source.read()
                    if not ret:
                        self.logger.warning("Failed to read frame from camera")
                        if reconnect_attempts < self.config["max_reconnect_attempts"]:
                            capture_source.release()
                            time.sleep(2)
                            capture_source = self.create_opencv_capture()
                            reconnect_attempts += 1
                            continue
                        else:
                            self.logger.error("Max reconnection attempts reached")
                            break
                    reconnect_attempts = 0
                    self.frame_count += 1
                    fps_counter += 1
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - fps_start_time >= 1.0:
                        self.current_fps = fps_counter
                        fps_counter = 0
                        fps_start_time = current_time
                    
                    if self.frame_count % self.frame_skip != 0:
                        with self.frame_lock:
                            self.last_frame = frame.copy()
                        continue
                    
                    self.frame_buffer.append(frame.copy())
                    results = self.model(frame, verbose=False)
                    annotated_frame, violations = self.process_detections(frame, results)
                    assigned_violations = self.assign_violation_ids(violations)
                    violations_to_save = self.should_save_violation(assigned_violations)
                    
                    for violation_id, violation_box in violations_to_save:
                        self.total_violations += 1
                        filename = self.save_violation_screenshot(annotated_frame, violation_box)
                        if filename:
                            self.logger.info(f"NO HELMET violation detected - ID: {violation_id}")
                            print(f"NO HELMET violation detected - ID: {violation_id}")
                    
                    camera_info = self.get_current_camera_info()
                    info_text = f"{camera_info['name']} | NO HELMET: {self.total_violations} | Active: {len(assigned_violations)}"
                    with self.frame_lock:
                        cv2.putText(self.last_frame, info_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(self.last_frame, f"FPS: {self.current_fps}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    time.sleep(0.033)
                except Exception as e:
                    self.logger.error(f"Error in detection loop: {e}")
                    print(f"Error in detection loop: {e}")
                    time.sleep(1)
                    continue
        finally:
            if capture_source:
                capture_source.release()
            self.is_running = False
            self.logger.info("Detection loop stopped")
            print("Detection loop stopped")

    def start_detection_thread(self):
        """Start detection in a separate thread"""
        if not self.is_running:
            self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.detection_thread.start()
            return True
        return False

    def stop(self):
        """Stop the detection system"""
        self.is_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)


# Flask Application
app = Flask(__name__)
detector = None

def generate_frames():
    """Generate frames for Flask streaming"""
    global detector
    while True:
        if detector and detector.is_running:
            with detector.frame_lock:
                if detector.last_frame is not None:
                    display_frame = cv2.resize(
                        detector.last_frame,
                        (detector.config["display_width"], detector.config["display_height"])
                    )
                    ret, buffer = cv2.imencode('.jpg', display_frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Generate a black frame when not running
            black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Detection Stopped", (500, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_cameras')
def get_cameras():
    """Get list of all cameras"""
    global detector
    if detector:
        return jsonify(detector.get_cameras_list())
    return jsonify([])

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    """Switch to a different camera"""
    global detector
    if detector:
        data = request.get_json()
        camera_id = data.get('camera_id')
        success, message = detector.switch_camera(camera_id)
        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "camera_name": detector.get_current_camera_info()["name"] if success else None
        })
    return jsonify({"status": "error", "message": "Detector not initialized"})

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start helmet detection"""
    global detector
    if detector:
        if detector.start_detection_thread():
            return jsonify({"status": "success", "message": "Detection started successfully"})
        else:
            return jsonify({"status": "error", "message": "Detection is already running"})
    return jsonify({"status": "error", "message": "Detector not initialized"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop helmet detection"""
    global detector
    if detector:
        detector.stop()
        return jsonify({"status": "success", "message": "Detection stopped"})
    return jsonify({"status": "error", "message": "Detector not initialized"})

@app.route('/reset_violations', methods=['POST'])
def reset_violations():
    """Reset violation counters"""
    global detector
    if detector:
        detector.reset_violations()
        return jsonify({"status": "success", "message": "Violation counters reset"})
    return jsonify({"status": "error", "message": "Detector not initialized"})

@app.route('/get_stats')
def get_stats():
    """Get current system statistics"""
    global detector
    if detector:
        return jsonify(detector.get_stats())
    return jsonify({"status": "error", "message": "Detector not initialized"})

@app.route('/get_violation_info')
def get_violation_info():
    """Get violation directory and file information"""
    global detector
    if detector:
        return jsonify(detector.get_violation_info())
    return jsonify({"status": "error", "message": "Detector not initialized"})

if __name__ == "__main__":
    print("Starting Helmet Detection System with Flask Web Interface...")

    try:
        detector = HelmetDetectionSystem()
        print("Detection system initialized successfully!")
        print("Web server starting...")

        # Render environment ke PORT ko use karo
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
        if detector:
            detector.stop()
    except Exception as e:
        print(f"System error: {e}")
        logging.error(f"System error: {e}")
        if detector:
            detector.stop()

    except Exception as e:
        print(f"System error: {e}")
        logging.error(f"System error: {e}")
        if detector:
            detector.stop()



