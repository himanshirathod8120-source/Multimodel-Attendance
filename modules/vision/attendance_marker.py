"""
attendance_marker.py
====================
Manages attendance marking based on sustained presence.
A student must be detected continuously for a configurable threshold
(default 30 seconds) before being marked as "present".
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class AttendanceRecord:
    """Tracks presence duration for a single student."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.first_seen: float = time.time()
        self.last_seen: float = time.time()
        self.total_presence_seconds: float = 0.0
        self.marked_present: bool = False
        self.mark_time: Optional[float] = None
        self.absence_count: int = 0

    def update_seen(self, timestamp: float):
        elapsed = timestamp - self.last_seen
        if elapsed < 5.0:  # Only count if < 5s gap (not a new detection)
            self.total_presence_seconds += elapsed
        self.last_seen = timestamp

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "first_seen": datetime.fromtimestamp(self.first_seen).isoformat(),
            "last_seen": datetime.fromtimestamp(self.last_seen).isoformat(),
            "total_presence_seconds": round(self.total_presence_seconds, 1),
            "marked_present": self.marked_present,
            "mark_time": datetime.fromtimestamp(self.mark_time).isoformat() if self.mark_time else None,
        }


class AttendanceMarker:
    """
    Marks attendance for students based on sustained presence.
    Thread-safe; call update() per frame with detected identities.
    """

    def __init__(self, config: dict, session_config: dict):
        self.presence_threshold = session_config.get("presence_threshold_seconds", 30)
        self.min_confidence = session_config.get("min_confidence", 0.60)
        self.log_dir = Path(session_config.get("log_dir", "data/logs"))
        self.mark_unknown = session_config.get("mark_unknown", True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._records: dict[str, AttendanceRecord] = {}
        self._session_start = time.time()
        self._session_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self.log_dir / f"attendance_{self._session_date}.json"
        logger.info(f"Attendance log: {self._log_path}")

    def update(self, identities: list[tuple[str, float]]) -> list[str]:
        """
        Update presence records from current frame identities.

        Args:
            identities: List of (name, confidence) from face recognition

        Returns:
            List of student names newly marked as present this call
        """
        now = time.time()
        newly_marked = []

        for name, confidence in identities:
            if not self.mark_unknown and name == "Unknown":
                continue
            if confidence < self.min_confidence and name != "Unknown":
                continue

            if name not in self._records:
                self._records[name] = AttendanceRecord(name)
                logger.info(f"New face detected: {name}")

            record = self._records[name]
            record.update_seen(now)

            if (not record.marked_present and
                    record.total_presence_seconds >= self.presence_threshold):
                record.marked_present = True
                record.mark_time = now
                newly_marked.append(name)
                logger.info(
                    f"✓ ATTENDANCE MARKED: {name} "
                    f"(after {record.total_presence_seconds:.0f}s presence)"
                )

        return newly_marked

    def get_summary(self) -> dict:
        """Return current attendance summary."""
        present = [r.to_dict() for r in self._records.values() if r.marked_present]
        absent = [r.to_dict() for r in self._records.values() if not r.marked_present]
        return {
            "session_date": self._session_date,
            "session_duration_seconds": round(time.time() - self._session_start, 1),
            "total_detected": len(self._records),
            "present_count": len(present),
            "present": present,
            "absent_or_insufficient": absent,
        }

    def save_log(self):
        """Save attendance log to JSON file."""
        summary = self.get_summary()
        with open(self._log_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Attendance saved → {self._log_path}")
        return self._log_path

    def get_present_students(self) -> list[str]:
        """Return list of marked-present student names."""
        return [name for name, r in self._records.items() if r.marked_present]

    def get_presence_progress(self) -> dict[str, float]:
        """
        Return {name: fraction_towards_threshold} for display.
        1.0+ means marked present.
        """
        progress = {}
        for name, record in self._records.items():
            progress[name] = min(
                1.0, record.total_presence_seconds / self.presence_threshold
            )
        return progress
