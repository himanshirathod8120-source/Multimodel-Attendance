"""
app.py
======
Flask web dashboard — robust version.
Works even when orchestrator is None (shows live camera feed + mock state).
"""

import sys
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ─────────────────────────────────────────────────────────────
_orchestrator = None
_cap = None
_lock = threading.Lock()
_frame_buffer: bytes = b""
_running = False
_capture_thread = None


def set_orchestrator(orch):
    global _orchestrator
    _orchestrator = orch


def start_background_capture(source=0):
    """
    Capture video in background thread.
    If orchestrator is set, runs _process_frame on each frame.
    Falls back to raw webcam feed if no orchestrator.
    """
    global _cap, _frame_buffer, _running, _capture_thread

    _running = True
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    _cap = cv2.VideoCapture(source)
    if not _cap.isOpened():
        logger.warning(f"Camera {source} not accessible — dashboard will show placeholder.")

    def _loop():
        global _frame_buffer
        while _running:
            try:
                if _cap and _cap.isOpened():
                    ret, frame = _cap.read()
                    if ret and frame is not None:
                        # Run AI pipeline if orchestrator ready
                        if _orchestrator:
                            try:
                                frame = _orchestrator._process_frame(frame)
                            except Exception as e:
                                logger.debug(f"Frame process error: {e}")
                        # Encode to JPEG
                        ok, buf = cv2.imencode(".jpg", frame,
                                              [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            with _lock:
                                _frame_buffer = buf.tobytes()
                else:
                    # Camera not available — make a placeholder frame
                    placeholder = _make_placeholder()
                    ok, buf = cv2.imencode(".jpg", placeholder)
                    if ok:
                        with _lock:
                            _frame_buffer = buf.tobytes()
            except Exception as e:
                logger.debug(f"Capture loop error: {e}")
            time.sleep(0.033)  # ~30 fps cap

    _capture_thread = threading.Thread(target=_loop, daemon=True, name="VideoCaptureThread")
    _capture_thread.start()
    return _capture_thread


def _make_placeholder() -> np.ndarray:
    """Generate a placeholder frame when no camera is available."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (25, 20, 45)  # Dark purple-ish background
    cv2.putText(frame, "No Camera Feed", (160, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (108, 92, 231), 2)
    cv2.putText(frame, "Dashboard is Active", (165, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 206, 201), 2)
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (270, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 180), 1)
    return frame


def _gen_frames():
    """MJPEG streaming generator."""
    while True:
        with _lock:
            buf = _frame_buffer
        if buf:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        time.sleep(0.033)


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    if _orchestrator:
        try:
            return jsonify(_orchestrator.get_state())
        except Exception as e:
            logger.error(f"get_state error: {e}")

    return jsonify({
        "class_engagement": 0,
        "student_engagements": {},
        "motion": {"is_active": False, "activity_score": 0},
        "audio": {"is_speech": False, "speech_ratio": 0},
        "attendance": {"present_count": 0, "total_detected": 0, "present": []},
        "suggestions": [],
        "alerts": [],
        "presence_progress": {},
        "engagement_averages": {},
        "enrolled_count": 0,
        "session_duration_min": 0,
    })


@app.route("/api/snapshots")
def api_snapshots():
    if _orchestrator:
        try:
            return jsonify(_orchestrator.engagement_engine.get_snapshots()[-200:])
        except Exception:
            pass
    return jsonify([])


@app.route("/api/save_report", methods=["POST"])
def api_save_report():
    if _orchestrator:
        try:
            _orchestrator._save_reports()
            return jsonify({"status": "saved"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No active session"}), 400


@app.route("/reports/<path:filename>")
def serve_report(filename):
    report_dir = Path(__file__).resolve().parents[2] / "data" / "reports"
    return send_from_directory(str(report_dir), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
