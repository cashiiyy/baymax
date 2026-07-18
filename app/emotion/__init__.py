"""BAYMAX AI – emotion package."""
from app.emotion.webcam_capture import WebcamCapture, VideoFrame
from app.emotion.face_detector import FaceDetector, FaceDetection
from app.emotion.embedder import FaceEmbedder
from app.emotion.deepface_engine import EmotionEngine, EmotionScore, EmotionResult, EMOTION_LABELS

__all__ = [
    "WebcamCapture", "VideoFrame", "FaceDetector", "FaceDetection",
    "FaceEmbedder", "EmotionEngine", "EmotionScore", "EmotionResult", "EMOTION_LABELS",
]
