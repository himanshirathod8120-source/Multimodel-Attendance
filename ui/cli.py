"""
cli.py
======
Command-line interface for the Multimodal Attendance & Engagement Agent.
Parses arguments and launches the appropriate mode.
"""

import argparse
import sys
import logging
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attendance-agent",
        description="Multimodal Attendance & Engagement Agent — AI-powered classroom intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live webcam session with web dashboard
  python main.py --mode live --dashboard

  # Process a recorded video file
  python main.py --mode file --input lecture.mp4

  # Live session, no display window (headless/server)
  python main.py --mode live --no-display

  # Enroll student faces interactively
  python main.py --enroll "John Doe"

  # Generate report from last session log
  python main.py --report
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=["live", "file"],
        default="live",
        help="Session mode: 'live' (webcam) or 'file' (recorded video). Default: live",
    )

    # Input
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to video file (required for --mode file). Can also be a camera index.",
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        default=None,
        help="Path to audio file to use alongside a video file input.",
    )

    # Display & Dashboard
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch the web dashboard (browser UI) instead of OpenCV window.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable all visual display (headless mode).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web dashboard (default: 5000).",
    )

    # Config
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML file. Default: config/settings.yaml",
    )

    # Utility commands
    parser.add_argument(
        "--enroll",
        type=str,
        metavar="STUDENT_NAME",
        help="Enroll a new student by capturing face images from webcam.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML/PDF report from the most recent session log.",
    )
    parser.add_argument(
        "--list-students",
        action="store_true",
        help="List all enrolled students.",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    return parser


def handle_enroll(name: str, config: dict):
    """Capture webcam images to enroll a new student."""
    import cv2
    known_dir = Path(config["face_recognition"]["known_faces_dir"]) / name
    known_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📷 Enrolling student: {name}")
    print("  Press SPACE to capture a photo (capture 5 images)")
    print("  Press Q to quit\n")

    cap = cv2.VideoCapture(0)
    count = 0
    target = 5

    while count < target:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, f"Capturing: {count}/{target} — SPACE to capture, Q to quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.imshow(f"Enrolling: {name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            img_path = known_dir / f"{name}_{count + 1}.jpg"
            cv2.imwrite(str(img_path), frame)
            print(f"  ✅ Captured image {count + 1}/{target}: {img_path}")
            count += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Enrolled {count} image(s) for '{name}' in {known_dir}")


def handle_list_students(config: dict):
    """List all enrolled students."""
    known_dir = Path(config["face_recognition"]["known_faces_dir"])
    if not known_dir.exists():
        print("No students enrolled yet.")
        return
    students = [d.name for d in known_dir.iterdir() if d.is_dir()]
    if not students:
        print("No students enrolled yet.")
        return
    print(f"\n👥 Enrolled students ({len(students)}):")
    for s in sorted(students):
        img_count = len(list((known_dir / s).glob("*.jpg")) +
                        list((known_dir / s).glob("*.png")))
        print(f"  • {s}  ({img_count} image{'s' if img_count != 1 else ''})")


def handle_report(config: dict):
    """Generate report from most recent session log."""
    import json
    log_dir = Path(config["attendance"]["log_dir"])
    logs = sorted(log_dir.glob("attendance_*.json"))
    if not logs:
        print("No session logs found.")
        return
    latest = logs[-1]
    print(f"📊 Generating report from: {latest}")

    with open(latest) as f:
        attendance_summary = json.load(f)

    from modules.analytics.session_analytics import SessionAnalytics
    analytics = SessionAnalytics(config["reports"])
    html_path = analytics.generate_html_report(
        attendance_summary=attendance_summary,
        engagement_snapshots=[],
        student_averages={},
        audio_summary={"total_frames_analyzed": 0, "speech_percentage": 0,
                       "avg_rms": 0, "avg_audio_score": 0},
        suggestions=[],
        session_duration_s=attendance_summary.get("session_duration_seconds", 0),
    )
    print(f"✅ Report saved → {html_path}")
    import webbrowser
    webbrowser.open(str(html_path))
