"""
BAYMAX AI – MediaPipe Face Detector
=====================================
Detects faces in video frames using MediaPipe Face Detection.
Returns bounding boxes and face region crops for downstream processing.

Usage:
    from app.emotion.face_detector import FaceDetector
    detector = FaceDetector()
    detections = detector.detect(frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FaceDetection:
    """
    A single detected face.

    Attributes:
        bbox:         Bounding box as (x, y, w, h) in pixel coordinates.
        confidence:   Detection confidence score (0–1).
        face_crop:    Cropped face image (BGR).
        landmarks:    Keypoint positions as [(x, y), ...] if available.
    """
    bbox: Tuple[int, int, int, int]     # x, y, w, h
    confidence: float
    face_crop: Optional[np.ndarray] = None
    landmarks: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


class FaceDetector:
    """
    MediaPipe-based face detector.

    Attributes:
        min_confidence: Minimum detection confidence to accept a face.
        model_selection: 0 for short-range (2m), 1 for full-range (5m).
        padding:        Fractional padding around detected face for cropping.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        model_selection: int = 0,
        padding: float = 0.2,
    ) -> None:
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self.padding = padding
        self._detector = None  # Lazy init
        log.info(
            "FaceDetector configured | min_confidence={} model={}",
            min_confidence,
            model_selection,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a BGR image frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            List of FaceDetection objects, ordered by confidence (desc).
        """
        self._ensure_detector()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._detector.process(rgb)  # type: ignore[union-attr]

        if not results.detections:
            return []

        detections: List[FaceDetection] = []
        for detection in results.detections:
            score = detection.score[0]
            if score < self.min_confidence:
                continue

            # Convert relative bbox to pixel coords
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * w))
            y = max(0, int(box.ymin * h))
            bw = min(int(box.width * w), w - x)
            bh = min(int(box.height * h), h - y)

            # Apply padding
            pad_x = int(bw * self.padding)
            pad_y = int(bh * self.padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            face_crop = frame[y1:y2, x1:x2].copy()

            # Extract landmarks
            landmarks = []
            for kp in detection.location_data.relative_keypoints:
                px, py = int(kp.x * w), int(kp.y * h)
                landmarks.append((px, py))

            detections.append(
                FaceDetection(
                    bbox=(x, y, bw, bh),
                    confidence=round(score, 4),
                    face_crop=face_crop if face_crop.size > 0 else None,
                    landmarks=landmarks,
                )
            )

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        log.debug("Faces detected: {} in {}x{} frame", len(detections), w, h)
        return detections

    def detect_primary(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Return only the most confident face detection.

        Args:
            frame: BGR image.

        Returns:
            Best FaceDetection or None.
        """
        detections = self.detect(frame)
        return detections[0] if detections else None

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[FaceDetection],
    ) -> np.ndarray:
        """
        Draw bounding boxes and scores on a frame copy.

        Args:
            frame:      Original BGR frame.
            detections: List of detections to draw.

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(
                annotated, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            label = f"Face {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector:
            self._detector.close()
            self._detector = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_detector(self) -> None:
        if self._detector is not None:
            return
        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence,
        )
        log.info("MediaPipe FaceDetector loaded")

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
