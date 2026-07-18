"""
BAYMAX AI – DeepFace Emotion Recognition Engine
=================================================
Runs DeepFace emotion analysis on face crops from the FaceDetector.
Implements temporal smoothing across frames to reduce jitter.

Emotions detected:
    angry, disgust, fear, happy, sad, surprise, neutral

Usage:
    from app.emotion.deepface_engine import EmotionEngine
    engine = EmotionEngine()
    result = engine.analyze_frame(frame)
    print(result.dominant_emotion)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from app.emotion.face_detector import FaceDetector, FaceDetection
from app.emotion.webcam_capture import WebcamCapture, VideoFrame
from app.utils.logger import get_logger

log = get_logger(__name__)

# All supported emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


@dataclass
class EmotionScore:
    """
    Emotion classification result for a single frame.

    Attributes:
        dominant_emotion:  Highest-scoring emotion label.
        scores:            Dict mapping emotion → confidence (0–100 scale from DeepFace).
        normalized_scores: Dict mapping emotion → confidence (0–1 scale).
        confidence:        Confidence of the dominant emotion (0–1).
        face_detected:     Whether a face was found in the frame.
        timestamp:         Unix timestamp of analysis.
    """
    dominant_emotion: str
    scores: Dict[str, float]
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    face_detected: bool = True
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        total = sum(self.scores.values()) or 1.0
        self.normalized_scores = {
            k: round(v / total, 4) for k, v in self.scores.items()
        }
        self.confidence = self.normalized_scores.get(self.dominant_emotion, 0.0)

    @property
    def is_negative(self) -> bool:
        return self.dominant_emotion in ("angry", "disgust", "fear", "sad")

    @property
    def is_positive(self) -> bool:
        return self.dominant_emotion == "happy"

    @property
    def is_distressed(self) -> bool:
        """Check if user appears distressed (high negative emotion confidence)."""
        distress_score = sum(
            self.normalized_scores.get(e, 0.0)
            for e in ("angry", "fear", "sad", "disgust")
        )
        return distress_score > 0.6

    def to_dict(self) -> Dict:
        return {
            "dominant_emotion": self.dominant_emotion,
            "confidence": self.confidence,
            "scores": self.normalized_scores,
            "face_detected": self.face_detected,
            "is_distressed": self.is_distressed,
        }


@dataclass
class EmotionResult:
    """
    Temporally-smoothed emotion result (across N frames).

    Attributes:
        current:    Latest raw emotion score.
        smoothed:   Smoothed scores averaged over window.
        frame_count: Number of frames in smoothing window.
    """
    current: EmotionScore
    smoothed: EmotionScore
    frame_count: int = 1


class EmotionEngine:
    """
    DeepFace-based emotion recognition with temporal smoothing.

    Processes video frames to detect the user's emotional state.
    Smoothing reduces per-frame jitter by averaging over a rolling window.

    Attributes:
        face_detector:      FaceDetector instance.
        smoothing_window:   Number of frames to average for smoothing.
        detector_backend:   DeepFace face detector backend.
        enforce_detection:  Whether to raise errors if no face found.
    """

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        smoothing_window: Optional[int] = None,
        detector_backend: Optional[str] = None,
        enforce_detection: bool = False,
    ) -> None:
        from config import settings

        self.face_detector = face_detector or FaceDetector()
        self.smoothing_window = smoothing_window or settings.EMOTION_SMOOTHING_FRAMES
        self.detector_backend = detector_backend or settings.DEEPFACE_BACKEND
        self.enforce_detection = enforce_detection

        self._history: Deque[EmotionScore] = deque(maxlen=self.smoothing_window)
        log.info(
            "EmotionEngine configured | backend={} smoothing={}",
            self.detector_backend,
            self.smoothing_window,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> Optional[EmotionResult]:
        """
        Analyze emotion from a BGR video frame.

        Steps:
            1. Detect primary face with MediaPipe
            2. Run DeepFace emotion analysis on face crop
            3. Apply temporal smoothing

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            EmotionResult, or None if no face detected.
        """
        # Step 1: Detect face
        face = self.face_detector.detect_primary(frame)
        if face is None or face.face_crop is None:
            log.debug("No face detected in frame")
            return None

        # Step 2: Run DeepFace on face crop
        raw_score = self._run_deepface(face.face_crop)
        if raw_score is None:
            return None

        # Step 3: Temporal smoothing
        self._history.append(raw_score)
        smoothed = self._compute_smoothed()

        return EmotionResult(
            current=raw_score,
            smoothed=smoothed,
            frame_count=len(self._history),
        )

    def analyze_image_bytes(self, image_bytes: bytes) -> Optional[EmotionResult]:
        """
        Analyze emotion from raw JPEG/PNG image bytes.

        Args:
            image_bytes: Image bytes (e.g. from API upload).

        Returns:
            EmotionResult or None.
        """
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            log.error("Failed to decode image bytes")
            return None
        return self.analyze_frame(frame)

    def run_continuous(
        self,
        webcam: WebcamCapture,
        callback=None,
        skip_frames: int = 5,
    ) -> None:
        """
        Run continuous emotion detection from a webcam stream.

        Args:
            webcam:      Running WebcamCapture instance.
            callback:    Called with EmotionResult for each analyzed frame.
            skip_frames: Only analyze every Nth frame to reduce CPU load.
        """
        frame_count = 0
        for video_frame in webcam.frames():
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            result = self.analyze_frame(video_frame.image)
            if result and callback:
                callback(result)

    def reset_history(self) -> None:
        """Clear the smoothing history buffer."""
        self._history.clear()

    # ── Private Methods ───────────────────────────────────────────────────────

    def _run_deepface(self, face_crop: np.ndarray) -> Optional[EmotionScore]:
        """Run DeepFace on a cropped face image."""
        try:
            from deepface import DeepFace

            # DeepFace expects BGR numpy array
            analysis = DeepFace.analyze(
                face_crop,
                actions=["emotion"],
                detector_backend="skip",  # Face already cropped, skip detection
                enforce_detection=self.enforce_detection,
                silent=True,
            )

            # DeepFace returns a list; take the first result
            if isinstance(analysis, list):
                analysis = analysis[0]

            emotion_scores: Dict[str, float] = analysis.get("emotion", {})
            dominant = analysis.get("dominant_emotion", "neutral")

            # Normalize keys to lowercase
            emotion_scores = {k.lower(): v for k, v in emotion_scores.items()}

            # Fill in any missing labels
            for label in EMOTION_LABELS:
                emotion_scores.setdefault(label, 0.0)

            return EmotionScore(
                dominant_emotion=dominant.lower(),
                scores=emotion_scores,
            )

        except ValueError as exc:
            log.debug("DeepFace ValueError (face too small?): {}", exc)
            return None
        except Exception as exc:
            log.warning("DeepFace analysis error: {}", exc)
            return None

    def _compute_smoothed(self) -> EmotionScore:
        """Average emotion scores over the history window."""
        if not self._history:
            return EmotionScore(
                dominant_emotion="neutral",
                scores={label: 0.0 for label in EMOTION_LABELS},
            )

        averaged: Dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}
        count = len(self._history)

        for score in self._history:
            for label in EMOTION_LABELS:
                averaged[label] += score.scores.get(label, 0.0)

        averaged = {k: v / count for k, v in averaged.items()}
        dominant = max(averaged, key=lambda k: averaged[k])

        return EmotionScore(
            dominant_emotion=dominant,
            scores=averaged,
        )
