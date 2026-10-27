"""
orchestrator.py
===============
Main agent loop: Perceive → Process → Fuse → Reason → Act.
Fully robust — gracefully handles missing ML packages and cameras.
"""

import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional

from modules.vision.face_detector import FaceDetector
from modules.vision.head_pose_estimator import HeadPoseEstimator, AttentionState
from modules.vision.movement_tracker import MovementTracker
from modules.vision.attendance_marker import AttendanceMarker
from modules.audio.audio_capture import AudioCapture
from modules.audio.audio_analyzer import AudioAnalyzer
from modules.analytics.engagement_engine import EngagementEngine
from modules.analytics.recommendation import RecommendationEngine
from modules.analytics.session_analytics import SessionAnalytics
from modules.agent.memory import AgentMemory

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central agent that coordinates all modules.
    Runs the perceive-reason-act loop for each video frame.
    All modules fail safely — the system continues if any module errors.
    """

    def __init__(self, config: dict):
        self.config = config
        self.memory = AgentMemory()
        self.running = False
        self._frame_count = 0
        self._suggestions: list[str] = []

        logger.info("Initializing modules...")

        # Vision modules
        try:
            self.face_detector = FaceDetector(config["face_recognition"])
        except Exception as e:
            logger.error(f"FaceDetector init failed: {e}")
            self.face_detector = None

        try:
            self.head_pose = HeadPoseEstimator(config["head_pose"])
        except Exception as e:
            logger.warning(f"HeadPoseEstimator init failed: {e}")
            self.head_pose = None

        try:
            self.motion_tracker = MovementTracker(config["motion"])
        except Exception as e:
            logger.warning(f"MovementTracker init failed: {e}")
            self.motion_tracker = None

        self.attendance = AttendanceMarker({}, config["attendance"])

        # Audio modules (optional)
        self.audio_enabled = config["audio"].get("enabled", True)
        self.audio_capture: Optional[AudioCapture] = None
        self.audio_analyzer: Optional[AudioAnalyzer] = None
        if self.audio_enabled:
            try:
                self.audio_capture = AudioCapture(config["audio"])
                self.audio_analyzer = AudioAnalyzer(config["audio"])
            except Exception as e:
                logger.warning(f"Audio init failed: {e} — running without audio.")
                self.audio_enabled = False

        # Analytics & AI
        self.engagement_engine = EngagementEngine(config["engagement"])
        self.recommendation = RecommendationEngine(config["ai"])
        self.session_analytics = SessionAnalytics(config["reports"])

        # Student enrollment
        if self.face_detector:
            logger.info("Enrolling student faces...")
            self.enrolled_faces = self.face_detector.enroll_students()
            logger.info(f"Enrolled {len(self.enrolled_faces)} student(s)")
        else:
            self.enrolled_faces = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_live(self, display: bool = True):
        """Run live session from webcam."""
        source = self.config["video"].get("source", 0)
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}. Running in no-camera mode.")
            # Still run the loop, just generate blank frames
            self._run_no_camera_loop(display)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["video"].get("frame_width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["video"].get("frame_height", 720))

        self._start_audio()
        self.running = True
        logger.info("Session started. Press 'q' to quit, 's' to save report.")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream.")
                    break
                annotated = self._process_frame(frame)
                if display:
                    cv2.imshow("Multimodal Attendance & Engagement Agent", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        self._save_reports()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.running = False
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self._stop_audio()
            self._save_reports()

    def _run_no_camera_loop(self, display: bool):
        """Run session loop with blank frames (no camera available)."""
        self._start_audio()
        self.running = True
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No Camera — Dashboard Active", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        logger.info("Running in no-camera mode. Dashboard will show mock data.")
        try:
            while self.running:
                self._process_frame(blank.copy())
                if display:
                    cv2.imshow("Multimodal Attendance & Engagement Agent", blank)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if display:
                cv2.destroyAllWindows()
            self._stop_audio()
            self._save_reports()

    def run_file(self, video_path: str, audio_path: Optional[str] = None, display: bool = False):
        """Process a recorded video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing file: {video_path} ({total} frames @ {fps:.0f}fps)")

        self.running = True
        frame_idx = 0
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = self._process_frame(frame)
                if display:
                    cv2.imshow("Processing...", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_idx += 1
                if frame_idx % 100 == 0:
                    pct = 100 * frame_idx // max(1, total)
                    logger.info(f"  Progress: {frame_idx}/{total} ({pct}%)")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self._save_reports()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the full pipeline on a single frame. Never raises."""
        self._frame_count += 1

        # ── PERCEIVE ─────────────────────────────────────────────────
        faces = []
        if self.face_detector:
            try:
                faces = self.face_detector.detect_faces(frame)
            except Exception as e:
                logger.debug(f"Face detect error: {e}")

        poses = []
        if self.head_pose:
            try:
                poses = self.head_pose.estimate(frame)
            except Exception as e:
                logger.debug(f"Head pose error: {e}")

        motion_result = {"motion_area": 0, "is_active": False, "activity_score": 0.5,
                         "time_since_motion": 0.0, "fg_mask": None}
        if self.motion_tracker:
            try:
                motion_result = self.motion_tracker.process_frame(frame)
            except Exception as e:
                logger.debug(f"Motion error: {e}")

        audio_result = {"audio_score": 0.0, "speech_ratio": 0.0, "is_speech": False}
        if self.audio_enabled and self.audio_capture and self.audio_analyzer:
            try:
                audio_frame = self.audio_capture.get_frame(timeout=0.01)
                if audio_frame is not None:
                    audio_result = self.audio_analyzer.process_frame(audio_frame)
            except Exception as e:
                logger.debug(f"Audio error: {e}")

        # ── IDENTIFY ─────────────────────────────────────────────────
        identities = []
        for face in faces:
            try:
                name, conf = self.face_detector.recognize_face(face, self.enrolled_faces)
            except Exception:
                name, conf = "Unknown", 0.0
            identities.append((name, conf))

        # ── ATTENDANCE ───────────────────────────────────────────────
        try:
            newly_marked = self.attendance.update(identities)
            for name in newly_marked:
                self.memory.log_event("attendance", {"student": name, "status": "marked_present"})
        except Exception as e:
            logger.debug(f"Attendance update error: {e}")

        # ── ENGAGEMENT FUSION ─────────────────────────────────────────
        engagements = {}
        for i, (name, conf) in enumerate(identities):
            if name == "Unknown":
                continue
            try:
                attn_score = poses[i]["attention_score"] if i < len(poses) else 0.5
                attn_state = poses[i]["attention"].value if i < len(poses) else "attentive"
                snap = self.engagement_engine.compute(
                    student_id=name,
                    attention_score=attn_score,
                    activity_score=motion_result["activity_score"],
                    audio_score=audio_result.get("audio_score", 0.0),
                    attention_state=attn_state,
                )
                engagements[name] = snap.engagement_score
            except Exception as e:
                logger.debug(f"Engagement compute error: {e}")

        class_engagement = self.engagement_engine.compute_class_engagement()

        # ── REASON ───────────────────────────────────────────────────
        try:
            anomalies = self.engagement_engine.detect_anomalies()
            for anomaly in anomalies:
                self.memory.add_alert(anomaly)

            suggestion = self.recommendation.generate(
                class_engagement=class_engagement,
                student_averages=self.engagement_engine.get_all_averages(),
                anomalies=anomalies,
                session_duration_min=self.memory.get_session_duration_minutes(),
            )
            if suggestion:
                self._suggestions.append(suggestion)
                self.memory.add_suggestion(suggestion)
                logger.info(f"[Suggestion] {suggestion}")
        except Exception as e:
            logger.debug(f"Reason error: {e}")

        # ── ACT: Update memory ──────────────────────────────────────
        self.memory.update_frame_state(
            frame_id=self._frame_count,
            faces=faces,
            identities=identities,
            poses=poses,
            motion=motion_result,
            audio=audio_result,
            engagements=engagements,
            class_engagement=class_engagement,
        )

        # ── ANNOTATE ─────────────────────────────────────────────────
        try:
            return self._annotate(frame, faces, identities, motion_result,
                                  audio_result, engagements, class_engagement)
        except Exception as e:
            logger.debug(f"Annotate error: {e}")
            return frame

    def _annotate(self, frame, faces, identities, motion, audio,
                  engagements, class_engagement) -> np.ndarray:
        """Draw annotations on the frame."""
        if self.face_detector:
            out = self.face_detector.draw_faces(frame, faces, identities)
        else:
            out = frame.copy()

        if self.motion_tracker:
            out = self.motion_tracker.draw_motion(out, motion)

        h, w = out.shape[:2]

        # Class engagement (top-right)
        eng_color = (0, 200, 80) if class_engagement >= 70 else \
                    (0, 180, 255) if class_engagement >= 40 else (0, 80, 220)
        label = f"Class Eng: {class_engagement:.0f}"
        cv2.rectangle(out, (w-260, 5), (w-5, 45), (20, 20, 40), -1)
        cv2.putText(out, label, (w-255, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eng_color, 2)

        # Per-face engagement scores
        for i, (name, _) in enumerate(identities):
            if name in engagements and i < len(faces):
                score = engagements[name]
                x, y, fw, fh = faces[i]["bbox"]
                ec = (0, 200, 80) if score >= 70 else (0, 180, 255) if score >= 40 else (0, 80, 220)
                cv2.putText(out, f"Eng:{score:.0f}", (x, y + fh + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ec, 2)

        # Audio badge
        if audio.get("is_speech"):
            cv2.rectangle(out, (5, h-40), (150, h-10), (20, 20, 40), -1)
            cv2.putText(out, "MIC Active", (10, h-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        # Latest suggestion bar
        if self.memory.latest_suggestion:
            text = self.memory.latest_suggestion[:95]
            cv2.rectangle(out, (0, h-38), (w, h), (20, 20, 50), -1)
            cv2.putText(out, text, (8, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)

        return out

    def _start_audio(self):
        if self.audio_enabled and self.audio_capture:
            try:
                self.audio_capture.start_microphone()
                logger.info("Microphone started.")
            except Exception as e:
                logger.warning(f"Audio start failed: {e}")

    def _stop_audio(self):
        if self.audio_capture:
            try:
                self.audio_capture.stop()
            except Exception:
                pass

    def _save_reports(self):
        """Save attendance log and generate session report."""
        try:
            logger.info("Saving session reports...")
            self.attendance.save_log()

            audio_summary = {"total_frames_analyzed": 0, "speech_percentage": 0.0,
                             "avg_rms": 0.0, "avg_audio_score": 0.0}
            if self.audio_analyzer:
                try:
                    audio_summary = self.audio_analyzer.get_session_summary()
                except Exception:
                    pass

            html_path = self.session_analytics.generate_html_report(
                attendance_summary=self.attendance.get_summary(),
                engagement_snapshots=self.engagement_engine.get_snapshots(),
                student_averages=self.engagement_engine.get_all_averages(),
                audio_summary=audio_summary,
                suggestions=self._suggestions,
                session_duration_s=self.memory.get_session_duration_seconds(),
            )

            if self.config["reports"].get("generate_pdf"):
                self.session_analytics.generate_pdf_report(html_path)

            logger.info(f"Reports saved → {html_path.parent}")
        except Exception as e:
            logger.error(f"Report save failed: {e}")

    def get_state(self) -> dict:
        """Return current state snapshot for the dashboard API (JSON-safe)."""
        try:
            snap = self.memory.get_state_snapshot()
            # Strip non-JSON-serializable objects (numpy arrays, etc.)
            motion = dict(snap.get("motion", {}))
            motion.pop("fg_mask", None)  # ndarray — not serializable
            snap["motion"] = motion

            return {
                **snap,
                "attendance": self.attendance.get_summary(),
                "presence_progress": self.attendance.get_presence_progress(),
                "engagement_averages": self.engagement_engine.get_all_averages(),
                "suggestions": self._suggestions[-5:],
                "alerts": self.memory.alerts[-5:],
                "enrolled_count": len(self.enrolled_faces),
            }
        except Exception as e:
            logger.error(f"get_state error: {e}")
            return {
                "class_engagement": 0, "student_engagements": {},
                "motion": {}, "audio": {},
                "attendance": {"present_count": 0, "total_detected": 0, "present": []},
                "suggestions": [], "alerts": [], "presence_progress": {},
                "engagement_averages": {}, "enrolled_count": 0,
            }
