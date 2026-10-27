"""
memory.py
=========
Session-level short-term memory for the agent.
Stores current state, event log, and recent context for reasoning.
"""

import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    timestamp: float
    event_type: str      # "attendance", "alert", "suggestion", "anomaly"
    payload: dict


class AgentMemory:
    """
    Maintains session state accessible across the agent loop iterations.
    """

    def __init__(self, max_events: int = 500):
        self._events: deque = deque(maxlen=max_events)
        self.session_start = time.time()

        # Current-frame state (updated every iteration)
        self.current_frame_id: int = 0
        self.current_faces: list = []
        self.current_identities: list = []
        self.current_poses: list = []
        self.current_motion: dict = {}
        self.current_audio: dict = {}
        self.current_engagements: dict = {}  # {student_id: score}
        self.class_engagement: float = 0.0
        self.latest_suggestion: Optional[str] = None
        self.suggestions_log: list = []
        self.alerts: list = []

    def log_event(self, event_type: str, payload: dict):
        """Record a notable event."""
        event = AgentEvent(
            timestamp=round(time.time() - self.session_start, 2),
            event_type=event_type,
            payload=payload,
        )
        self._events.append(event)
        logger.debug(f"[Event] {event_type}: {payload}")

    def update_frame_state(
        self,
        frame_id: int,
        faces: list,
        identities: list,
        poses: list,
        motion: dict,
        audio: dict,
        engagements: dict,
        class_engagement: float,
    ):
        """Update state with latest frame analysis results."""
        self.current_frame_id = frame_id
        self.current_faces = faces
        self.current_identities = identities
        self.current_poses = poses
        # Strip numpy arrays that are not JSON-serializable
        clean_motion = {k: v for k, v in motion.items() if k != "fg_mask"}
        self.current_motion = clean_motion
        self.current_audio = audio
        self.current_engagements = engagements
        self.class_engagement = class_engagement

    def add_suggestion(self, suggestion: str):
        self.latest_suggestion = suggestion
        self.suggestions_log.append({
            "time": round(time.time() - self.session_start, 2),
            "text": suggestion,
        })

    def add_alert(self, alert: dict):
        self.alerts.append({**alert, "time": round(time.time() - self.session_start, 2)})

    def get_session_duration_seconds(self) -> float:
        return round(time.time() - self.session_start, 1)

    def get_session_duration_minutes(self) -> float:
        return round(self.get_session_duration_seconds() / 60, 2)

    def get_event_log(self) -> list:
        return [
            {
                "timestamp": e.timestamp,
                "type": e.event_type,
                "payload": e.payload,
            }
            for e in self._events
        ]

    def get_state_snapshot(self) -> dict:
        return {
            "session_duration_min": self.get_session_duration_minutes(),
            "frame_id": self.current_frame_id,
            "class_engagement": self.class_engagement,
            "student_engagements": self.current_engagements,
            "latest_suggestion": self.latest_suggestion,
            "audio": self.current_audio,
            "motion": self.current_motion,
        }
