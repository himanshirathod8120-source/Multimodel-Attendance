"""
voice_activity.py
=================
Voice Activity Detection. Uses webrtcvad if available,
else falls back to simple RMS-energy threshold (no extra deps).
"""

import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

try:
    import webrtcvad as _webrtcvad
    _VAD_AVAILABLE = True
    logger.info("webrtcvad available.")
except ImportError:
    _VAD_AVAILABLE = False
    logger.info("webrtcvad not installed — using RMS energy VAD fallback.")


class VoiceActivityDetector:
    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.chunk_ms = config.get("chunk_duration_ms", 30)
        self.aggressiveness = config.get("vad_aggressiveness", 2)
        self._history: deque = deque(maxlen=100)
        self._vad = None

        if _VAD_AVAILABLE:
            try:
                self._vad = _webrtcvad.Vad(self.aggressiveness)
            except Exception as e:
                logger.warning(f"webrtcvad init failed: {e}")

    def is_speech(self, frame: np.ndarray) -> bool:
        if self._vad is not None:
            try:
                raw = frame.astype(np.int16).tobytes()
                result = self._vad.is_speech(raw, self.sample_rate)
                self._history.append(1 if result else 0)
                return result
            except Exception:
                pass
        # RMS fallback
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        is_sp = rms > 500  # ~empirical threshold for speech
        self._history.append(1 if is_sp else 0)
        return is_sp

    def speech_ratio(self, window_size: int = 33) -> float:
        if not self._history:
            return 0.0
        recent = list(self._history)[-window_size:]
        return round(sum(recent) / len(recent), 3)

    def is_currently_speaking(self, threshold: float = 0.4) -> bool:
        return self.speech_ratio() >= threshold

    def reset_history(self):
        self._history.clear()
