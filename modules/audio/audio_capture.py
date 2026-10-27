"""
audio_capture.py
================
Handles real-time microphone capture and audio file reading.
Gracefully disables itself if PyAudio is not installed.
"""

import logging
import threading
import queue
import time
import numpy as np
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    import pyaudio
    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False
    logger.info("PyAudio not installed — audio capture disabled (mock mode).")


class AudioCapture:
    """
    Captures audio from microphone or reads from file.
    Falls back to silent mock if PyAudio not available.
    """

    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.chunk_ms = config.get("chunk_duration_ms", 30)
        self.device_index = config.get("device_index", None)
        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)
        self.available = _PYAUDIO_AVAILABLE

        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._stream = None
        self._pa = None

    def start_microphone(self):
        if not self.available:
            logger.warning("PyAudio not available — running without audio.")
            return
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index,
                stream_callback=self._mic_callback,
            )
            self._running = True
            self._stream.start_stream()
            logger.info(f"Microphone started (rate={self.sample_rate}Hz)")
        except Exception as e:
            logger.warning(f"Microphone failed to open: {e} — continuing without audio.")
            self.available = False

    def _mic_callback(self, in_data, frame_count, time_info, status):
        if self._running:
            try:
                self._queue.put_nowait(in_data)
            except queue.Full:
                pass
        return (None, pyaudio.paContinue)

    def get_frame(self, timeout: float = 0.01) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            raw = self._queue.get(timeout=timeout)
            return np.frombuffer(raw, dtype=np.int16)
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
        logger.info("Audio capture stopped.")
