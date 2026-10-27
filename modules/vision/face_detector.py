"""
face_detector.py
================
Face detection using OpenCV's built-in Haar Cascade (zero extra deps).
Optional: upgrades to DeepFace if installed.
Recognition uses simple LBPH face recognizer (OpenCV built-in).
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try DeepFace — use it if available, else fall back to OpenCV
try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
    logger.info("DeepFace available — using for recognition.")
except ImportError:
    _DEEPFACE_AVAILABLE = False
    logger.info("DeepFace not installed — using OpenCV LBPH recognizer (built-in).")


class FaceDetector:
    """
    Detects and recognizes faces.
    Primary: OpenCV Haar Cascade + LBPH (always available).
    Optional upgrade: DeepFace if installed.
    """

    def __init__(self, config: dict):
        self.config = config
        self.known_faces_dir = Path(config.get("known_faces_dir", "data/known_faces"))
        self.threshold = config.get("threshold", 0.4)

        # Load Haar cascade (ships with opencv-python)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            logger.error("Could not load Haar cascade XML.")

        # LBPH recognizer for offline recognition
        self._recognizer = cv2.face.LBPHFaceRecognizer_create() if hasattr(cv2, 'face') else None
        self._label_map: dict[int, str] = {}  # label_id -> name
        self._is_trained = False

        logger.info("FaceDetector initialized (OpenCV Haar + LBPH)")

    def enroll_students(self) -> dict:
        """
        Load known student faces from known_faces/<StudentName>/*.jpg
        Trains LBPH recognizer on enrollment images.
        Returns: {name: True} to indicate enrolled
        """
        enrolled = {}
        if not self.known_faces_dir.exists():
            logger.warning(f"known_faces dir not found: {self.known_faces_dir}")
            return enrolled

        faces_data = []
        labels = []
        label_id = 0

        for student_dir in sorted(self.known_faces_dir.iterdir()):
            if not student_dir.is_dir():
                continue
            name = student_dir.name
            images_found = 0
            for img_path in student_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Detect face in enrollment image
                face_rects = self._cascade.detectMultiScale(img, 1.1, 5, minSize=(30, 30))
                if len(face_rects) > 0:
                    x, y, w, h = face_rects[0]
                    face_roi = cv2.resize(img[y:y+h, x:x+w], (100, 100))
                else:
                    face_roi = cv2.resize(img, (100, 100))
                faces_data.append(face_roi)
                labels.append(label_id)
                images_found += 1

            if images_found > 0:
                self._label_map[label_id] = name
                enrolled[name] = True
                label_id += 1
                logger.info(f"Enrolled '{name}' with {images_found} image(s).")

        if faces_data and self._recognizer:
            try:
                self._recognizer.train(faces_data, np.array(labels))
                self._is_trained = True
                logger.info(f"LBPH recognizer trained on {len(faces_data)} images, {len(enrolled)} students.")
            except Exception as e:
                logger.warning(f"LBPH training failed: {e}")

        logger.info(f"Total enrolled students: {len(enrolled)}")
        return enrolled

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in frame using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces = []
        for (x, y, w, h) in rects:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            faces.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": 0.9,
                "face_img": face_roi_color,
                "face_gray": face_roi_gray,
            })
        return faces

    def recognize_face(self, face_img, enrolled: dict) -> tuple[str, float]:
        """
        Identify a face using LBPH recognizer.
        Returns (name, confidence).
        """
        if not enrolled or not self._is_trained or self._recognizer is None:
            return "Unknown", 0.0
        try:
            if isinstance(face_img, dict):
                gray = face_img.get("face_gray")
            else:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            if gray is None or gray.size == 0:
                return "Unknown", 0.0
            gray = cv2.resize(gray, (100, 100))
            label, dist = self._recognizer.predict(gray)
            name = self._label_map.get(label, "Unknown")
            # LBPH: lower distance = better match (0=perfect, >80=poor)
            confidence = max(0.0, 1.0 - dist / 100.0)
            if dist > 80:
                return "Unknown", confidence
            return name, round(confidence, 3)
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return "Unknown", 0.0

    def draw_faces(self, frame: np.ndarray, faces: list[dict],
                   identities: Optional[list] = None) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        out = frame.copy()
        for i, face in enumerate(faces):
            x, y, w, h = face["bbox"]
            color = (0, 255, 100)
            label = ""
            if identities and i < len(identities):
                name, conf = identities[i]
                label = f"{name} ({conf:.0%})"
                color = (0, 200, 255) if name == "Unknown" else (0, 255, 100)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            if label:
                # Background for text readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(out, (x, y - th - 10), (x + tw + 4, y), color, -1)
                cv2.putText(out, label, (x + 2, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return out
