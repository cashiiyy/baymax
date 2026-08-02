"""
BAYMAX AI — Multimodal Routes
================================
REST endpoints that expose AI Engine 2's multimodal capabilities to the
frontend (and any other consumers) via AI Engine 1.

All routes degrade gracefully: if AE2 is unreachable they return a clear
``{"available": false}`` response rather than crashing.

Prefix: /multimodal
"""

from __future__ import annotations

import io
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from app.utils.ae2_client import ae2_client
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/multimodal", tags=["Multimodal (AI Engine 2)"])


# ── Response Models ───────────────────────────────────────────────────────────

class AE2Unavailable(BaseModel):
    available: bool = False
    detail: str = "AI Engine 2 is currently unreachable. Multimodal features are degraded."


class VoiceProfile(BaseModel):
    name: str
    language: Optional[str] = None
    description: Optional[str] = None
    is_cloned: Optional[bool] = None


# ── Health / Status ───────────────────────────────────────────────────────────

@router.get("/ae2-status")
async def ae2_status():
    """
    Probe AI Engine 2's health and component readiness.
    Safe to call from the frontend for the status badge.
    """
    healthy = await ae2_client.health()
    if not healthy:
        return {"available": False, "status": "offline"}

    status = await ae2_client.status() or {}
    return {"available": True, "status": "online", **status}


# ── Text-to-Speech ────────────────────────────────────────────────────────────

@router.post("/tts")
async def tts(
    text: str = Form(...),
    voice: str = Form("default"),
    language: str = Form("en"),
):
    """
    Synthesize speech via AI Engine 2 (XTTS v2).
    Returns base64-encoded WAV + duration.
    """
    result = await ae2_client.tts(text=text, voice=voice, language=language)
    if result is None:
        return AE2Unavailable()
    return {"available": True, **result}


@router.get("/voices")
async def list_voices():
    """List TTS voice profiles available on AI Engine 2."""
    voices = await ae2_client.get_voices()
    return {"available": True, "voices": voices}


# ── Speech-to-Text ────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
):
    """
    Transcribe an audio file via AI Engine 2 (Faster-Whisper).
    Returns text, language, word_count, has_speech, and word-level timestamps.
    """
    audio_bytes = await file.read()
    result = await ae2_client.transcribe(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio.webm",
        language=language,
    )
    if result is None:
        return AE2Unavailable()
    return {"available": True, **result}


# ── OCR ───────────────────────────────────────────────────────────────────────

@router.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form("eng"),
):
    """
    Extract text and structured fields from a medical document or image
    via AI Engine 2 (Tesseract + PyMuPDF).

    Returns:
      - raw_text
      - document_type (prescription | lab_report | medical_record | ...)
      - extracted_fields [{field, value, confidence}]
      - overall_confidence
    """
    file_bytes = await file.read()
    result = await ae2_client.ocr(
        file_bytes=file_bytes,
        filename=file.filename or "document.jpg",
        lang=lang,
    )
    if result is None:
        return AE2Unavailable()
    return {"available": True, **result}


# ── Vision ────────────────────────────────────────────────────────────────────

@router.post("/vision")
async def vision(file: UploadFile = File(...)):
    """
    Analyse an image for face detection, emotion context, and person presence
    via AI Engine 2 (OpenCV + MediaPipe).

    Returns:
      - faces (list of detected face bounding boxes)
      - emotions [{label, confidence}]
      - person_detected (bool)
      - lighting_assessment
      - observations (list of plain-language observations)

    NOTE: All outputs are observational — not diagnostic.
    """
    image_bytes = await file.read()
    result = await ae2_client.vision(
        image_bytes=image_bytes,
        filename=file.filename or "frame.jpg",
    )
    if result is None:
        return AE2Unavailable()
    return {"available": True, **result}


# ── Document Parser ───────────────────────────────────────────────────────────

@router.post("/document")
async def document(file: UploadFile = File(...)):
    """
    Full structured document parsing via AI Engine 2 (PyMuPDF).
    Handles PDF, image, TXT, and Markdown files.

    Returns parsed sections, tables, metadata, and extracted fields.
    """
    file_bytes = await file.read()
    result = await ae2_client.document(
        file_bytes=file_bytes,
        filename=file.filename or "document.pdf",
    )
    if result is None:
        return AE2Unavailable()
    return {"available": True, **result}


# ── Combined Analysis ─────────────────────────────────────────────────────────

@router.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    tasks: str = Form("ocr"),
):
    """
    Run multiple AI Engine 2 tasks on one file in a single parallel call.

    ``tasks`` is a comma-separated list:
      e.g. "ocr,vision" | "ocr,transcribe" | "ocr,vision,transcribe"

    This is the most efficient path when the Backend needs multiple analyses
    on one uploaded file.
    """
    file_bytes = await file.read()
    task_list  = [t.strip() for t in tasks.split(",") if t.strip()]

    result = await ae2_client.analyse(
        file_bytes=file_bytes,
        filename=file.filename or "upload",
        tasks=task_list,
    )
    if result is None:
        return AE2Unavailable()
    return {"available": True, **result}
