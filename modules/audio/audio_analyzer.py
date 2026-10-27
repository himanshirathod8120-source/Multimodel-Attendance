"""
audio_analyzer.py
=================
Analyzes audio signals to extract participation and noise metrics.
Combines VAD, RMS energy, and speech density into an audio engagement score.
"""

import numpy as np
import logging
import time
from collections import deque
from .voice_activity import VoiceActivityDetector

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Computes audio-based engagement metrics:
    - RMS energy level
    - Speech density (VAD-based)
    - Audio participation score
    """

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.chunk_ms = config.get("chunk_duration_ms", 30)
        self.vad = VoiceActivityDetector(config)

        self._rms_history: deque = deque(maxlen=200)
        self._speech_history: deque = deque(maxlen=200)
        self._timeline: list = []  # [{time, rms, speech, audio_score}]
        self._session_start = time.time()

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one audio chunk.

        Args:
            frame: int16 numpy array

        Returns:
            dict: rms, rms_db, is_speech, speech_ratio, audio_score
        """
        # RMS energy
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        rms_db = 20 * np.log10(rms + 1e-9)  # dB, avoid log(0)
        self._rms_history.append(rms)

        # VAD
        is_speech = self.vad.is_speech(frame)
        self._speech_history.append(1 if is_speech else 0)

        speech_ratio = self.vad.speech_ratio()
        audio_score = self._compute_audio_score(rms, speech_ratio)

        result = {
            "rms": round(rms, 2),
            "rms_db": round(rms_db, 2),
            "is_speech": is_speech,
            "speech_ratio": speech_ratio,
            "audio_score": audio_score,
            "timestamp": round(time.time() - self._session_start, 2),
        }

        # Store timeline entry every 5 seconds
        if not self._timeline or result["timestamp"] - self._timeline[-1]["timestamp"] >= 5:
            self._timeline.append({
                "time": result["timestamp"],
                "rms": result["rms"],
                "speech_ratio": speech_ratio,
                "audio_score": audio_score,
            })

        return result

    def _compute_audio_score(self, rms: float, speech_ratio: float) -> float:
        """
        Combine RMS energy and speech ratio into a [0, 1] audio score.
        Normalize RMS relative to recent history.
        """
        # Normalize RMS relative to recent max
        recent_rms = list(self._rms_history)
        if recent_rms and max(recent_rms) > 0:
            rms_norm = min(1.0, rms / (max(recent_rms) + 1e-9))
        else:
            rms_norm = 0.0

        # Weighted combination
        score = 0.4 * rms_norm + 0.6 * speech_ratio
        return round(min(1.0, score), 3)

    def get_average_audio_score(self) -> float:
        """Average audio score over the entire session."""
        if not self._speech_history:
            return 0.0
        return round(float(np.mean(list(self._speech_history))), 3)

    def get_timeline(self) -> list:
        return self._timeline

    def get_session_summary(self) -> dict:
        speech = list(self._speech_history)
        rms_vals = list(self._rms_history)
        return {
            "total_frames_analyzed": len(speech),
            "speech_percentage": round(sum(speech) / max(1, len(speech)) * 100, 1),
            "avg_rms": round(float(np.mean(rms_vals)) if rms_vals else 0.0, 2),
            "avg_audio_score": self.get_average_audio_score(),
        }
