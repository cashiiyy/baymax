"""
BAYMAX AI – STT Routes
========================
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from app.stt.whisper_engine import WhisperEngine
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/stt", tags=["Speech Recognition"])

# Instantiate engine (lazy loads model on first request)
whisper_engine = WhisperEngine()


class STTResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration_s: float


@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file (WAV/MP3)."""
    import shutil
    import tempfile
    from pathlib import Path

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save uploaded file to temp path
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = whisper_engine.transcribe_file(tmp_path)
        return STTResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration_s=result.duration_s,
        )
    except Exception as exc:
        log.error("Transcription failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
