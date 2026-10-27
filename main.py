"""
main.py
=======
Entry point for the Multimodal Attendance & Engagement Agent.
Run with: python main.py --dashboard
"""

import sys
import os
import logging
import yaml
import threading
import webbrowser
import time
from pathlib import Path

# Project root on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ["urllib3", "PIL", "tensorflow", "deepface", "mediapipe",
                  "absl", "cv2", "werkzeug"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict):
    dirs = [
        config["face_recognition"]["known_faces_dir"],
        config["attendance"]["log_dir"],
        config["reports"]["output_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def print_banner():
    print("=" * 60)
    print("  Multimodal Attendance & Engagement Agent")
    print("  AI-Powered Classroom Intelligence System")
    print("=" * 60)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="attendance-agent",
        description="Multimodal Attendance & Engagement Agent"
    )
    parser.add_argument("--mode", choices=["live", "file"], default="live")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Video file path or camera index")
    parser.add_argument("--dashboard", "-d", action="store_true",
                        help="Launch web dashboard (http://127.0.0.1:5000)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — no OpenCV window")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", "-c", type=str, default="config/settings.yaml")
    parser.add_argument("--enroll", type=str, metavar="NAME",
                        help="Enroll a student via webcam")
    parser.add_argument("--list-students", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="Generate report from last session")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("main")
    print_banner()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # CLI overrides
    if args.input:
        config["video"]["source"] = args.input
    config["dashboard"]["port"] = args.port

    ensure_dirs(config)

    # ── Utility commands ─────────────────────────────────────────────
    if args.enroll:
        _enroll_student(args.enroll, config)
        return

    if args.list_students:
        _list_students(config)
        return

    if args.report:
        _generate_report(config)
        return

    # ── Init orchestrator ─────────────────────────────────────────────
    logger.info("Starting Orchestrator...")
    try:
        from modules.agent.orchestrator import Orchestrator
        orch = Orchestrator(config)
    except Exception as e:
        logger.error(f"Orchestrator failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Dashboard mode ────────────────────────────────────────────────
    if args.dashboard:
        from ui.dashboard.app import app as flask_app, set_orchestrator, start_background_capture

        set_orchestrator(orch)
        source = config["video"].get("source", 0)
        host = config["dashboard"].get("host", "127.0.0.1")
        port = config["dashboard"].get("port", 5000)
        url = f"http://{host}:{port}"

        logger.info(f"Starting video capture (source={source})...")
        start_background_capture(source)

        # Start audio
        if config["audio"].get("enabled", True) and orch.audio_capture:
            try:
                orch.audio_capture.start_microphone()
            except Exception as e:
                logger.warning(f"Audio not started: {e}")

        # Open browser
        def _open():
            time.sleep(2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

        logger.info(f"Dashboard running at: {url}")
        logger.info("Press Ctrl+C to stop.")

        flask_app.run(host=host, port=port, debug=False,
                      use_reloader=False, threaded=True)

    # ── Live OpenCV window mode ───────────────────────────────────────
    elif args.mode == "live":
        orch.run_live(display=not args.no_display)

    # ── File processing mode ──────────────────────────────────────────
    elif args.mode == "file":
        if not args.input:
            logger.error("--input is required for --mode file")
            sys.exit(1)
        orch.run_file(args.input, display=not args.no_display)

    logger.info("Session ended.")


# ── Utility helpers ───────────────────────────────────────────────────

def _enroll_student(name: str, config: dict):
    import cv2
    known_dir = Path(config["face_recognition"]["known_faces_dir"]) / name
    known_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEnrolling: {name}")
    print("Press SPACE to capture (need 5 images), Q to quit\n")
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, f"Capture {count}/5 — SPACE=capture Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.imshow(f"Enroll: {name}", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = known_dir / f"{name}_{count+1}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  Saved: {path}")
            count += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nEnrolled {count} image(s) for '{name}' → {known_dir}")


def _list_students(config: dict):
    known_dir = Path(config["face_recognition"]["known_faces_dir"])
    if not known_dir.exists():
        print("No students enrolled.")
        return
    students = [d for d in known_dir.iterdir() if d.is_dir()]
    if not students:
        print("No students enrolled.")
        return
    print(f"\nEnrolled students ({len(students)}):")
    for s in sorted(students):
        imgs = list(s.glob("*.jpg")) + list(s.glob("*.png"))
        print(f"  • {s.name}  ({len(imgs)} image(s))")


def _generate_report(config: dict):
    import json
    log_dir = Path(config["attendance"]["log_dir"])
    logs = sorted(log_dir.glob("attendance_*.json"))
    if not logs:
        print("No session logs found.")
        return
    latest = logs[-1]
    print(f"Generating report from: {latest}")
    with open(latest) as f:
        att = json.load(f)
    from modules.analytics.session_analytics import SessionAnalytics
    analytics = SessionAnalytics(config["reports"])
    html = analytics.generate_html_report(
        attendance_summary=att,
        engagement_snapshots=[],
        student_averages={},
        audio_summary={"total_frames_analyzed": 0, "speech_percentage": 0,
                       "avg_rms": 0, "avg_audio_score": 0},
        suggestions=[],
        session_duration_s=att.get("session_duration_seconds", 0),
    )
    print(f"Report → {html}")
    import webbrowser
    webbrowser.open(str(html))


if __name__ == "__main__":
    main()
