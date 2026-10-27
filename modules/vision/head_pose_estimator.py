"""
head_pose_estimator.py
======================
Estimates head orientation using MediaPipe FaceMesh if available,
else uses a simple heuristic based on Haar landmark positions.
"""

import cv2
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AttentionState(Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    ABSENT = "absent"


# 3D reference model points
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

_MP_INDICES = [1, 152, 263, 33, 287, 57]

try:
    import mediapipe as mp
    # mediapipe 1.1+ changed API — solutions may not exist
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _mp_face_mesh = mp.solutions.face_mesh
        _MEDIAPIPE_AVAILABLE = True
        logger.info("MediaPipe FaceMesh available.")
    else:
        _MEDIAPIPE_AVAILABLE = False
        logger.info("MediaPipe installed but solutions API unavailable — using heuristic fallback.")
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.info("MediaPipe not installed — using heuristic head pose estimation.")


class HeadPoseEstimator:
    def __init__(self, config: dict):
        self.yaw_range = config.get("attention_yaw_range", [-25, 25])
        self.pitch_range = config.get("attention_pitch_range", [-20, 20])
        self._face_mesh = None

        if _MEDIAPIPE_AVAILABLE:
            try:
                self._face_mesh = _mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                logger.info("MediaPipe FaceMesh initialized.")
            except Exception as e:
                logger.warning(f"FaceMesh init failed: {e}")
                self._face_mesh = None

        # Fallback: eye cascade for simple gaze estimation
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def estimate(self, frame: np.ndarray) -> list[dict]:
        if self._face_mesh is not None:
            return self._mediapipe_estimate(frame)
        return self._heuristic_estimate(frame)

    def _mediapipe_estimate(self, frame: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        poses = []
        if not results.multi_face_landmarks:
            return poses

        h, w = frame.shape[:2]
        focal = w
        cam_matrix = np.array([[focal, 0, w/2],[0, focal, h/2],[0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark
            img_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in _MP_INDICES], dtype=np.float64)
            ok, rot_vec, _ = cv2.solvePnP(_MODEL_POINTS, img_pts, cam_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
            pitch = float(np.degrees(np.arctan2(-rot_mat[2,0], sy)))
            yaw   = float(np.degrees(np.arctan2(rot_mat[1,0], rot_mat[0,0])))
            roll  = float(np.degrees(np.arctan2(rot_mat[2,1], rot_mat[2,2])))
            attention = self._classify(pitch, yaw)
            poses.append({
                "pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2),
                "attention": attention,
                "attention_score": self._score(pitch, yaw),
            })
        return poses

    def _heuristic_estimate(self, frame: np.ndarray) -> list[dict]:
        """Simple heuristic: if eye detected → attentive, else distracted."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self._eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        num_faces_gray, _, _ = cv2.split(frame) if frame.ndim == 3 else (frame, frame, frame)

        # Estimate one pose entry per detected eye pair (rough)
        results = []
        # Just return one "attentive" record if eyes detected
        if len(eyes) > 0:
            results.append({
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "attention": AttentionState.ATTENTIVE,
                "attention_score": 0.85,
            })
        return results

    def _classify(self, pitch, yaw) -> AttentionState:
        if self.yaw_range[0] <= yaw <= self.yaw_range[1] and \
           self.pitch_range[0] <= pitch <= self.pitch_range[1]:
            return AttentionState.ATTENTIVE
        return AttentionState.DISTRACTED

    def _score(self, pitch, yaw) -> float:
        yaw_max = max(abs(self.yaw_range[0]), abs(self.yaw_range[1]))
        pitch_max = max(abs(self.pitch_range[0]), abs(self.pitch_range[1]))
        ys = max(0.0, 1.0 - abs(yaw) / (yaw_max * 2))
        ps = max(0.0, 1.0 - abs(pitch) / (pitch_max * 2))
        return round((ys + ps) / 2.0, 3)

    def draw_pose(self, frame: np.ndarray, poses: list[dict]) -> np.ndarray:
        out = frame.copy()
        for i, pose in enumerate(poses):
            state = pose["attention"].value.upper()
            score = pose["attention_score"]
            color = (0, 255, 100) if state == "ATTENTIVE" else (0, 80, 255)
            cv2.putText(out, f"Face {i+1}: {state} ({score:.2f})",
                        (10, 30 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()
