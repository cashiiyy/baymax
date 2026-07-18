"""
BAYMAX AI – Face Embedder
==========================
Generates 128-dimensional face embeddings using the face_recognition library.
Embeddings can be used for face identification and user tracking across sessions.

Usage:
    from app.emotion.embedder import FaceEmbedder
    embedder = FaceEmbedder()
    embedding = embedder.embed(face_crop)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


class FaceEmbedder:
    """
    128-d face embedding extractor using face_recognition (dlib under the hood).

    Embeddings are suitable for:
        - Face identity verification (compare Euclidean distance)
        - User recognition across sessions (store in vector DB)

    Threshold for same-person match: distance < 0.6
    """

    DISTANCE_THRESHOLD = 0.6  # Below this = same person

    def __init__(self, model: str = "large") -> None:
        """
        Args:
            model: 'small' (fast) or 'large' (more accurate). Default: 'large'.
        """
        self.model = model
        log.info("FaceEmbedder initialized | model={}", model)

    def embed(self, face_image: np.ndarray) -> Optional[List[float]]:
        """
        Generate a 128-d embedding for a face crop.

        Args:
            face_image: BGR or RGB numpy array of a cropped face.

        Returns:
            128-dimensional embedding as List[float], or None if embedding fails.
        """
        try:
            import face_recognition

            # face_recognition expects RGB
            if face_image.shape[-1] == 3:
                rgb = face_image[:, :, ::-1]  # BGR → RGB
            else:
                rgb = face_image

            # Resize to minimum acceptable size
            h, w = rgb.shape[:2]
            if h < 20 or w < 20:
                log.debug("Face crop too small for embedding: {}x{}", w, h)
                return None

            encodings = face_recognition.face_encodings(
                rgb,
                known_face_locations=[(0, w, h, 0)],  # Use full image as face
                model=self.model,
            )

            if not encodings:
                log.debug("No encoding generated from face crop")
                return None

            return encodings[0].tolist()

        except Exception as exc:
            log.warning("Face embedding error: {}", exc)
            return None

    def embed_batch(
        self,
        face_images: List[np.ndarray],
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of face crops.

        Args:
            face_images: List of BGR/RGB face arrays.

        Returns:
            List of embeddings (None for failed images).
        """
        return [self.embed(img) for img in face_images]

    @staticmethod
    def distance(emb1: List[float], emb2: List[float]) -> float:
        """
        Compute Euclidean distance between two 128-d face embeddings.

        Args:
            emb1: First embedding.
            emb2: Second embedding.

        Returns:
            Euclidean distance (lower = more similar).
        """
        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.linalg.norm(a - b))

    def is_same_person(
        self,
        emb1: List[float],
        emb2: List[float],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine whether two embeddings belong to the same person.

        Args:
            emb1:      First embedding.
            emb2:      Second embedding.
            threshold: Distance threshold (default: 0.6).

        Returns:
            True if embeddings are likely the same person.
        """
        thr = threshold or self.DISTANCE_THRESHOLD
        dist = self.distance(emb1, emb2)
        log.debug("Face distance={:.4f} threshold={}", dist, thr)
        return dist < thr
