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

# All supported emotion labels (including tensed and stressed)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "tensed", "stressed"]


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
        # Map fear / tension scores to tensed if high
        if "fear" in self.scores and "tensed" not in self.scores:
            self.scores["tensed"] = self.scores["fear"] * 0.8
        total = sum(self.scores.values()) or 1.0
        self.normalized_scores = {
            k: round(v / total, 4) for k, v in self.scores.items()
        }
        self.confidence = self.normalized_scores.get(self.dominant_emotion, 0.0)

    @property
    def is_negative(self) -> bool:
        return self.dominant_emotion in ("angry", "disgust", "fear", "sad", "tensed", "stressed")

    @property
    def is_positive(self) -> bool:
        return self.dominant_emotion == "happy"

    @property
    def is_distressed(self) -> bool:
        """Check if user appears distressed (high negative emotion confidence)."""
        distress_score = sum(
            self.normalized_scores.get(e, 0.0)
            for e in ("angry", "fear", "sad", "disgust", "tensed", "stressed")
        )
        return distress_score > 0.35

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
            3. If face crop fails, fall back to full-image DeepFace analysis
            4. Apply temporal smoothing

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            EmotionResult, or None if no face detected.
        """
        raw_score = None

        # Step 1: Try MediaPipe crop → DeepFace
        try:
            face = self.face_detector.detect_primary(frame)
            if face is not None and face.face_crop is not None:
                raw_score = self._run_deepface(face.face_crop)
        except Exception:
            pass

        # Step 2: Fallback — run DeepFace on the full image
        if raw_score is None:
            log.debug("MediaPipe crop failed or returned no result; trying full-image DeepFace analysis")
            raw_score = self._run_deepface_full_image(frame)

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

    def _get_hf_pipeline(self):
        """Lazy-load HuggingFace emotion classification pipeline (cached after first load)."""
        if not hasattr(self, "_hf_pipe") or self._hf_pipe is None:
            try:
                from transformers import pipeline
                # dima806/facial_emotions_image_detection: public ResNet model trained on FER2013
                # Labels: angry, disgust, fear, happy, neutral, sad, surprise
                self._hf_pipe = pipeline(
                    "image-classification",
                    model="dima806/facial_emotions_image_detection",
                    top_k=7,
                )
                log.info("HuggingFace facial emotion model loaded")
            except Exception as exc:
                log.warning("HuggingFace emotion model unavailable: {}", exc)
                self._hf_pipe = None
        return self._hf_pipe

    def _scores_from_hf(self, hf_result: list) -> Optional[EmotionScore]:
        """Convert HuggingFace pipeline output to EmotionScore."""
        if not hf_result:
            return None
        emotion_scores: Dict[str, float] = {}
        for item in hf_result:
            label = item["label"].lower()
            score = float(item["score"])
            emotion_scores[label] = score
        # Map fear → tensed / stressed
        if "fear" in emotion_scores:
            emotion_scores["tensed"] = emotion_scores["fear"] * 0.8
            emotion_scores["stressed"] = emotion_scores["fear"] * 0.6
        for label in EMOTION_LABELS:
            emotion_scores.setdefault(label, 0.0)
        dominant = max(emotion_scores, key=lambda k: emotion_scores[k])
        return EmotionScore(
            dominant_emotion=dominant,
            scores=emotion_scores,
            face_detected=True,
        )

    def _run_deepface(self, face_crop: np.ndarray) -> Optional[EmotionScore]:
        """Run emotion analysis on a cropped face image using HuggingFace ViT."""
        try:
            from PIL import Image
            pipe = self._get_hf_pipeline()
            if pipe is None:
                return None
            rgb = face_crop[:, :, ::-1]  # BGR → RGB
            pil_img = Image.fromarray(rgb)
            result = pipe(pil_img)
            return self._scores_from_hf(result)
        except Exception as exc:
            log.debug("HuggingFace face crop analysis failed: {}", exc)
            return None

    def _run_deepface_full_image(self, frame: np.ndarray) -> Optional[EmotionScore]:
        """Run emotion analysis on the full image using HuggingFace ViT + OpenCV Haar fallback."""
        # 1. Try HuggingFace on full image
        try:
            from PIL import Image
            pipe = self._get_hf_pipeline()
            if pipe is not None:
                rgb = frame[:, :, ::-1]
                pil_img = Image.fromarray(rgb)
                result = pipe(pil_img)
                score = self._scores_from_hf(result)
                if score:
                    return score
        except Exception as exc:
            log.debug("HuggingFace full-image analysis failed: {}", exc)

        # 2. OpenCV Haar cascade — face detected, but return neutral
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                emotion_scores = {label: 0.0 for label in EMOTION_LABELS}
                emotion_scores["neutral"] = 1.0
                return EmotionScore(dominant_emotion="neutral", scores=emotion_scores, face_detected=True)
        except Exception as exc:
            log.debug("Haar cascade fallback failed: {}", exc)

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
