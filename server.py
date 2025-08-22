from flask import Flask, request, jsonify, Response
import subprocess
import threading
import cv2
import os
import signal

app = Flask(__name__)

# Global variables
process = None
active_camera = None
violation_count = 0
is_running = False

# ---- Start detection (run train.py) ----
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global process, is_running
    if is_running:
        return jsonify({"success": False, "message": "Detection already running"})
    try:
        # Run train.py in background
        process = subprocess.Popen(["python", "train.py"])
        is_running = True
        return jsonify({"success": True, "message": "Detection started"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ---- Stop detection ----
@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global process, is_running
    if process:
        try:
            os.kill(process.pid, signal.SIGTERM)
            process = None
            is_running = False
            return jsonify({"success": True, "message": "Detection stopped"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    else:
        return jsonify({"success": False, "message": "No process running"})

# ---- Switch camera ----
@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global active_camera
    data = request.get_json()
    camera_id = data.get("camera_id")
    if camera_id:
        active_camera = camera_id
        return jsonify({"success": True, "camera_name": camera_id})
    else:
        return jsonify({"success": False, "message": "Invalid camera ID"})

# ---- Status ----
@app.route('/status', methods=['GET'])
def status():
    global is_running, active_camera, violation_count
    return jsonify({
        "status": "Running" if is_running else "Stopped",
        "is_running": is_running,
        "active_camera": active_camera or "None",
        "camera_name": active_camera or "None",
        "violation_count": violation_count
    })

# ---- Video feed ----
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Dummy camera for demo
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
