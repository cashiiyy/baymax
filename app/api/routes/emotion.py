"""
BAYMAX AI – Emotion Routes
============================
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict

from app.emotion.deepface_engine import EmotionEngine
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/emotion", tags=["Emotion Recognition"])

# Instantiating with short window for single-shot API checks
emotion_engine = EmotionEngine(smoothing_window=1)


class EmotionResponse(BaseModel):
    dominant_emotion: str
    confidence: float
    scores: Dict[str, float]
    is_distressed: bool


@router.post("/detect", response_model=EmotionResponse)
async def detect_emotion(file: UploadFile = File(...)):
    """Detect emotion from an uploaded image file (JPEG/PNG)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        contents = await file.read()
        result = emotion_engine.analyze_image_bytes(contents)

        if not result:
            raise HTTPException(status_code=400, detail="No face detected in image")

        score = result.current
        return EmotionResponse(
            dominant_emotion=score.dominant_emotion,
            confidence=score.confidence,
            scores=score.normalized_scores,
            is_distressed=score.is_distressed,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Emotion detection failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))
