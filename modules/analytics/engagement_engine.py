"""
engagement_engine.py
====================
Fuses vision (attention, motion) and audio signals into a unified
engagement score per student and class-wide.
"""

import time
import logging
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EngagementSnapshot:
    timestamp: float
    student_id: str
    attention_score: float    # 0-1 from head pose
    activity_score: float     # 0-1 from motion
    audio_score: float        # 0-1 from audio
    engagement_score: float   # 0-100 composite
    attention_state: str      # "attentive" | "distracted" | "absent"


class EngagementEngine:
    """
    Computes engagement scores from multimodal signals.

    Formula:
        engagement = 0.50 * attention_score
                   + 0.30 * activity_score
                   + 0.20 * audio_score
        (scaled 0-100)
    """

    WEIGHTS = {
        "attention": 0.50,
        "activity":  0.30,
        "audio":     0.20,
    }

    def __init__(self, config: dict):
        self.update_interval = config.get("update_interval_seconds", 5)
        alert_threshold = config.get("engagement_alert_threshold", 40)

        # Custom weights if configured
        if "attention_weight" in config:
            self.WEIGHTS = {
                "attention": config.get("attention_weight", 0.50),
                "activity":  config.get("activity_weight", 0.30),
                "audio":     config.get("audio_weight", 0.20),
            }

        # Per-student rolling score history (last 60 readings ≈ 5 min)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self._snapshots: list[EngagementSnapshot] = []
        self._last_update: dict[str, float] = {}
        self._class_timeline: list[dict] = []
        self._session_start = time.time()

    def compute(
        self,
        student_id: str,
        attention_score: float,
        activity_score: float,
        audio_score: float,
        attention_state: str = "attentive",
    ) -> EngagementSnapshot:
        """
        Compute engagement for one student at current moment.

        Args:
            student_id: Student name/ID
            attention_score: 0.0-1.0 from head pose
            activity_score: 0.0-1.0 from motion
            audio_score: 0.0-1.0 from audio
            attention_state: string label

        Returns:
            EngagementSnapshot
        """
        raw = (
            self.WEIGHTS["attention"] * attention_score +
            self.WEIGHTS["activity"]  * activity_score  +
            self.WEIGHTS["audio"]     * audio_score
        )
        score = round(min(100.0, max(0.0, raw * 100)), 1)

        snap = EngagementSnapshot(
            timestamp=round(time.time() - self._session_start, 2),
            student_id=student_id,
            attention_score=round(attention_score, 3),
            activity_score=round(activity_score, 3),
            audio_score=round(audio_score, 3),
            engagement_score=score,
            attention_state=attention_state,
        )
        self._history[student_id].append(score)
        self._snapshots.append(snap)
        return snap

    def compute_class_engagement(self) -> float:
        """Average engagement across all tracked students (last reading)."""
        if not self._history:
            return 0.0
        latest = [h[-1] for h in self._history.values() if h]
        return round(float(np.mean(latest)), 1)

    def get_student_trend(self, student_id: str) -> list[float]:
        """Return engagement score history for a student."""
        return list(self._history.get(student_id, []))

    def get_student_average(self, student_id: str) -> float:
        hist = self._history.get(student_id, [])
        if not hist:
            return 0.0
        return round(float(np.mean(list(hist))), 1)

    def get_all_averages(self) -> dict[str, float]:
        return {sid: self.get_student_average(sid) for sid in self._history}

    def get_snapshots(self) -> list[dict]:
        return [
            {
                "timestamp": s.timestamp,
                "student_id": s.student_id,
                "engagement_score": s.engagement_score,
                "attention_score": s.attention_score,
                "activity_score": s.activity_score,
                "audio_score": s.audio_score,
                "attention_state": s.attention_state,
            }
            for s in self._snapshots
        ]

    def detect_anomalies(self) -> list[dict]:
        """
        Detect engagement drops: students below 30 consistently.
        Returns list of alerts.
        """
        alerts = []
        for sid, hist in self._history.items():
            recent = list(hist)[-6:] if len(hist) >= 6 else list(hist)
            if recent and np.mean(recent) < 30:
                alerts.append({
                    "student": sid,
                    "avg_engagement": round(float(np.mean(recent)), 1),
                    "alert": "Sustained low engagement detected",
                })
        return alerts
