"""
BAYMAX AI — Multimodal Routes
================================
REST endpoints that expose AI Engine 2's multimodal capabilities to the
frontend (and any other consumers) via AI Engine 1.

All routes are fully self-healing: if AE2 is unreachable they seamlessly fall
back to local AI Engine 1 ML pipelines (Voicebox, Faster-Whisper, PyMuPDF,
and HuggingFace Emotion ViT) so user features never degrade or crash.

Prefix: /multimodal
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from pydantic import BaseModel

from app.utils.ae2_client import ae2_client
from app.tts.voicebox_engine import voicebox_engine
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/multimodal", tags=["Multimodal"])


# ── Response Models ───────────────────────────────────────────────────────────

class AE2Unavailable(BaseModel):
    available: bool = False
    detail: str = "Multimodal service degraded. Falling back to local offline capabilities."


class VoiceProfile(BaseModel):
    name: str
    language: Optional[str] = None
    description: Optional[str] = None
    is_cloned: Optional[bool] = None


# ── Health / Status ───────────────────────────────────────────────────────────

@router.get("/ae2-status")
async def ae2_status():
    """
    Probe AI Engine 2's health (BioMistral LLM on :1234 and Multimodal perception on :8001).
    Safe to call from the frontend for the status badge.
    """
    biomistral_online = False
    try:
        from app.llm.biomistral_engine import BioMistralEngine
        bio = BioMistralEngine()
        biomistral_online = await bio.is_healthy()
    except Exception:
        pass

    healthy = await ae2_client.health()
    status = await ae2_client.status() if healthy else {}

    is_online = biomistral_online or healthy
    return {
        "available": is_online,
        "status": "online" if is_online else "offline",
        "biomistral_online": biomistral_online,
        "multimodal_online": healthy,
        **(status or {}),
    }


# ── Text-to-Speech ────────────────────────────────────────────────────────────

@router.post("/tts")
async def tts(
    request: Request,
    text: Optional[str] = Form(None),
    voice: Optional[str] = Form("Baymax"),
    language: Optional[str] = Form("en"),
):
    """
    Synthesize speech via Voicebox TTS (Baymax profile) or AI Engine 2 (XTTS v2).
    Accepts both JSON payloads and multipart Form data.
    """
    req_text = text
    req_voice = voice
    req_lang = language

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            req_text = body.get("text", req_text)
            req_voice = body.get("voice", req_voice)
            req_lang = body.get("language", req_lang)
        except Exception:
            pass

    if not req_text or not req_text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    target_voice = req_voice if (req_voice and req_voice != "default") else "Baymax"

    # 1. Try Voicebox TTS
    try:
        res = await voicebox_engine.generate_speech(
            text=req_text,
            profile_name_or_id=target_voice,
            language=req_lang or "en",
        )
        if res.get("success") and res.get("audio_base64"):
            return {"available": True, **res}
    except Exception as exc:
        log.warning("Voicebox TTS unavailable in multimodal route: {}", exc)

    # 2. Fall back to AE2
    result = await ae2_client.tts(text=req_text, voice=target_voice, language=req_lang or "en")
    if result is not None and result.get("available") is not False:
        return {"available": True, **result}

    return AE2Unavailable()


@router.get("/voices")
async def list_voices():
    """List TTS voice profiles available on Voicebox and AI Engine 2."""
    voices = []
    try:
        vb_profiles = await voicebox_engine.get_profiles()
        for p in vb_profiles:
            voices.append({
                "name": p.get("name"),
                "language": p.get("language", "en"),
                "description": p.get("personality") or p.get("description"),
                "is_cloned": p.get("voice_type") == "cloned",
            })
    except Exception:
        pass

    if not any(v.get("name", "").lower() == "baymax" for v in voices):
        voices.insert(0, {
            "name": "Baymax",
            "language": "en",
            "description": "Baymax Voice Profile",
            "is_cloned": True,
        })

    try:
        ae2_voices = await ae2_client.get_voices()
        if ae2_voices:
            voices.extend(ae2_voices)
    except Exception:
        pass

    return {"available": True, "voices": voices}


# ── Speech-to-Text ────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
):
    """
    Transcribe an audio file via AI Engine 2 (Faster-Whisper) or local AI Engine 1.
    Decodes WebM, Opus, Ogg, WAV, and MP3 via PyAV.
    """
    audio_bytes = await file.read()
    filename = file.filename or "audio.webm"

    # 1. Try AI Engine 2
    try:
        result = await ae2_client.transcribe(
            audio_bytes=audio_bytes,
            filename=filename,
            language=language,
        )
        if result is not None and result.get("available") is not False:
            return {"available": True, **result}
    except Exception as exc:
        log.warning("AE2 transcribe proxy unavailable ({}), falling back to local Faster-Whisper...", exc)

    # 2. Local fallback using Faster-Whisper + PyAV
    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        from app.stt.whisper_engine import WhisperEngine
        engine = WhisperEngine()
        res = engine.transcribe_file(tmp_path)
        return {
            "available": True,
            "text": res.text,
            "language": res.language,
            "confidence": res.confidence,
            "duration_seconds": res.duration_s,
            "has_speech": not res.is_empty,
            "word_count": len(res.text.split()) if res.text else 0,
            "engine": "local_faster_whisper",
        }
    except Exception as w_err:
        log.error("Local whisper transcription failed: {}", w_err)
        return AE2Unavailable()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── OCR / Document Parsing ─────────────────────────────────────────────────────

@router.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form("eng"),
):
    """
    Extract text and structured clinical fields from medical documents or PDFs.
    Uses AI Engine 2 when online, or local PyMuPDF clinical parser when offline.
    """
    file_bytes = await file.read()
    filename = file.filename or "document.pdf"

    # 1. Try AE2 if online
    try:
        result = await ae2_client.ocr(
            file_bytes=file_bytes,
            filename=filename,
            lang=lang,
        )
        if result is not None and result.get("available") is not False:
            return {"available": True, **result}
    except Exception as exc:
        log.warning("AE2 OCR proxy unavailable ({}), falling back to local medical document parser...", exc)

    # 2. Local fallback using PyMuPDF clinical parser
    try:
        from app.ocr.medical_document_parser import medical_document_parser
        doc_result = medical_document_parser.parse_document_bytes(file_bytes, filename=filename)
        return doc_result
    except Exception as doc_err:
        log.error("Local medical document parsing error: {}", doc_err)
        return AE2Unavailable()


@router.post("/document")
async def document(file: UploadFile = File(...)):
    """
    Full structured document parsing for PDFs and clinical records.
    """
    file_bytes = await file.read()
    filename = file.filename or "document.pdf"

    # 1. Try AE2 if online
    try:
        result = await ae2_client.document(
            file_bytes=file_bytes,
            filename=filename,
        )
        if result is not None and result.get("available") is not False:
            return {"available": True, **result}
    except Exception as exc:
        log.warning("AE2 document proxy unavailable ({}), falling back to local parser...", exc)

    # 2. Local fallback
    try:
        from app.ocr.medical_document_parser import medical_document_parser
        doc_result = medical_document_parser.parse_document_bytes(file_bytes, filename=filename)
        return doc_result
    except Exception as doc_err:
        log.error("Local document parsing error: {}", doc_err)
        return AE2Unavailable()


# ── Vision ────────────────────────────────────────────────────────────────────

@router.post("/vision")
async def vision(file: UploadFile = File(...)):
    """
    Analyse an image for face detection, emotion context, and person presence.
    Runs AI Engine 2 when online or local OpenCV + HuggingFace ViT emotion engine when offline.
    """
    image_bytes = await file.read()
    filename = file.filename or "frame.jpg"

    # 1. Try AE2 if online
    try:
        result = await ae2_client.vision(
            image_bytes=image_bytes,
            filename=filename,
        )
        if result is not None and result.get("available") is not False:
            return {"available": True, **result}
    except Exception as exc:
        log.warning("AE2 vision proxy unavailable ({}), falling back to local emotion engine...", exc)

    # 2. Local fallback using local EmotionEngine & FaceDetector
    try:
        from app.emotion.deepface_engine import EmotionEngine
        from app.emotion.face_detector import FaceDetector
        import cv2
        import numpy as np

        engine = EmotionEngine()
        detector = FaceDetector()

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces_detected = []
        if frame is not None:
            raw_faces = detector.detect(frame)
            for f in raw_faces:
                bx, by, bw, bh = f.bbox
                faces_detected.append({
                    "x": bx, "y": by,
                    "w": bw, "h": bh,
                    "confidence": f.confidence,
                })

        emotion_res = engine.analyze_image_bytes(image_bytes)
        emotions_list = []
        dominant = "neutral"
        person = len(faces_detected) > 0 or (emotion_res is not None and emotion_res.face_detected)

        if emotion_res:
            dominant = emotion_res.dominant_emotion
            person = True
            for k, v in emotion_res.smoothed.normalized_scores.items():
                emotions_list.append({"label": k, "confidence": round(float(v), 3)})
            emotions_list.sort(key=lambda x: x["confidence"], reverse=True)

        lighting = "good"
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray))
            if mean_val < 45:
                lighting = "dim"
            elif mean_val > 210:
                lighting = "overexposed"

        obs = []
        if person:
            obs.append(f"Person detected with dominant emotion '{dominant}'")
        else:
            obs.append("No clear face detected")
        if lighting != "good":
            obs.append(f"Lighting condition: {lighting}")

        return {
            "available": True,
            "person_detected": person,
            "faces": faces_detected,
            "emotions": emotions_list,
            "dominant_emotion": dominant,
            "lighting_assessment": lighting,
            "observations": obs,
            "provider": "local_emotion_engine",
        }
    except Exception as local_err:
        log.error("Local vision analysis error: {}", local_err)
        return AE2Unavailable()


# ── Combined Analysis ─────────────────────────────────────────────────────────

@router.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    tasks: str = Form("ocr"),
):
    """
    Run multiple analysis tasks on one file.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    try:
        result = await ae2_client.analyse(
            file_bytes=file_bytes,
            filename=filename,
            tasks=task_list,
        )
        if result is not None and result.get("available") is not False:
            return {"available": True, **result}
    except Exception:
        pass

    # Fallback to local parsing
    from app.ocr.medical_document_parser import medical_document_parser
    return medical_document_parser.parse_document_bytes(file_bytes, filename=filename)
