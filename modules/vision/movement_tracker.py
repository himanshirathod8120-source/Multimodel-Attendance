"""
movement_tracker.py
===================
Detects and quantifies body movement/activity using background subtraction.
Provides an activity score per frame window.
"""

import cv2
import numpy as np
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class MovementTracker:
    """
    Uses background subtraction to detect movement in the scene.
    Computes an activity score based on motion magnitude over a rolling window.
    """

    def __init__(self, config: dict):
        self.movement_threshold = config.get("movement_threshold", 500)
        self.inactivity_window = config.get("inactivity_window_seconds", 60)
        method = config.get("background_subtractor", "MOG2")

        if method == "KNN":
            self._subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        else:
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False, varThreshold=50
            )

        self._motion_history: deque = deque(maxlen=300)  # rolling ~30s at 10fps
        self._last_motion_time = time.time()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Analyze movement in a single frame.

        Args:
            frame: BGR image

        Returns:
            dict with keys: motion_area, is_active, activity_score, time_since_motion
        """
        fg_mask = self._subtractor.apply(frame)
        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.dilate(fg_mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)

        is_active = motion_area > self.movement_threshold
        if is_active:
            self._last_motion_time = time.time()

        self._motion_history.append(1 if is_active else 0)
        activity_score = self._compute_activity_score()
        time_since_motion = time.time() - self._last_motion_time

        return {
            "motion_area": motion_area,
            "is_active": is_active,
            "activity_score": activity_score,
            "time_since_motion": round(time_since_motion, 1),
            "fg_mask": fg_mask,
        }

    def _compute_activity_score(self) -> float:
        """
        Activity score = fraction of recent frames with motion, in [0, 1].
        Weighted: recent frames matter more.
        """
        if not self._motion_history:
            return 0.0
        history = list(self._motion_history)
        n = len(history)
        # Linear weights: more recent = higher weight
        weights = np.linspace(0.5, 1.0, n)
        weighted_sum = float(np.dot(history, weights))
        max_sum = float(np.sum(weights))
        return round(weighted_sum / max_sum, 3)

    def draw_motion(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Overlay activity info on frame."""
        out = frame.copy()
        score = result["activity_score"]
        active_str = "ACTIVE" if result["is_active"] else "STILL"
        color = (0, 200, 80) if result["is_active"] else (60, 60, 200)

        h, w = frame.shape[:2]
        cv2.putText(out, f"Motion: {active_str}  Score: {score:.2f}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out
