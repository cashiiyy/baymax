"""
B.A.Y.M.A.X. Backend Server
============================
Acts as the **orchestration hub** between the browser, AI Engine 1 (reasoning),
and AI Engine 2 (multimodal perception).

Architecture:
  Browser / App
      ↕ HTTP
  AI Engine 1  (100.108.247.7:8000) — this file
      ↕ DB / Memory / Reasoning
  Backend Server (100.89.251.123:8000)
      ↕ httpx proxy
  AI Engine 2  (100.86.102.107:8001)

AI Engine 2 is NEVER imported as a Python module here — it is a
separate machine reached via HTTP over Tailscale.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env configuration
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)


import os
import uuid
from typing import Optional, List, Dict, Any

import httpx
from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database.db import get_db, engine, Base
from backend.database.models import User, Conversation
from backend.memory.memory_manager import MemoryManager
from backend.logging.logger import get_logger

from ai_engine_1.api.routes import router as engine1_router
from ai_engine_1.pipeline.reasoning_pipeline import reasoning_pipeline
from app.stt.whisper_engine import WhisperEngine

logger = get_logger("baymax-backend")

# Instantiate AI Engine 1 local Whisper engine
whisper_engine_ae1 = WhisperEngine()

# Singleton emotion engine — initialized once at startup to avoid per-request failures
_local_emotion_engine = None
def _get_emotion_engine():
    global _local_emotion_engine
    if _local_emotion_engine is None:
        try:
            from app.emotion.deepface_engine import EmotionEngine
            _local_emotion_engine = EmotionEngine(smoothing_window=3)
            logger.info("Local EmotionEngine singleton initialized")
        except Exception as exc:
            logger.warning(f"Local EmotionEngine initialization failed: {exc}")
    return _local_emotion_engine

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# ── AI Engine 2 Configuration ─────────────────────────────────────────────────
# Read from env; falls back to the confirmed Tailscale IP.
AE2_BASE_URL = os.getenv("AE2_BASE_URL", "http://100.79.169.64:8001").rstrip("/")
AE2_TIMEOUT  = float(os.getenv("AE2_TIMEOUT", "30"))

# In-memory registry so AE2 can self-register its dynamic IP at startup
_ae2_registry: Dict[str, str] = {"url": AE2_BASE_URL}

# Cached AE2 online status — updated by /api/health probes, not blocking /health
_ae2_status_cache: Dict[str, Any] = {"online": False, "last_checked": 0}


def _ae2_url() -> str:
    """Return the currently registered AE2 base URL."""
    return _ae2_registry.get("url", AE2_BASE_URL)


async def _proxy_to_ae2(
    path: str,
    *,
    files: dict | None = None,
    data: dict | None = None,
    json_body: dict | None = None,
    method: str = "POST",
) -> dict:
    """
    Forward a request to AI Engine 2 and return parsed JSON.
    Raises HTTPException(502) if AE2 is unreachable.
    """
    url = f"{_ae2_url()}{path}"
    correlation_id = str(uuid.uuid4())
    headers = {"X-Correlation-ID": correlation_id}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=AE2_TIMEOUT,
                                                            write=AE2_TIMEOUT, pool=5.0)) as client:
            if method == "GET":
                r = await client.get(url, headers=headers)
            elif files:
                r = await client.post(url, headers=headers, files=files, data=data or {})
            else:
                r = await client.post(url, headers=headers, json=json_body or {})
            r.raise_for_status()
            return r.json()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning(f"AE2 unreachable [{path}]: {exc}")
        raise HTTPException(status_code=502, detail="AI Engine 2 is currently unavailable.")
    except httpx.HTTPStatusError as exc:
        logger.error(f"AE2 HTTP error [{path}] {exc.response.status_code}: {exc.response.text}")
        raise HTTPException(status_code=502,
                            detail=f"AI Engine 2 returned {exc.response.status_code}")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="B.A.Y.M.A.X. Backend Orchestrator",
    description=(
        "Central gateway & orchestrator for the B.A.Y.M.A.X. multimodal medical assistant. "
        "Routes reasoning to AI Engine 1 and multimodal processing to AI Engine 2."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include AI Engine 1 REST endpoints
app.include_router(engine1_router)

# Static directory setup
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: int = 1
    query: str


class ChatResponse(BaseModel):
    user_id: int
    query: str
    response: str
    confidence: Optional[Dict[str, Any]] = None
    risk: Optional[str] = "low"


class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    language: str = "en"
    stream: bool = False


class AERegistrationPayload(BaseModel):
    engine: str           # e.g. "ai_engine_2"
    url: str              # e.g. "http://100.86.102.107:8001"
    version: Optional[str] = None
    capabilities: Optional[List[str]] = None


# ── UI Entrypoint ─────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/ui")
def serve_web_interface():
    """Serves the BAYMAX Web Interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "B.A.Y.M.A.X. Backend Server Online — UI not found."}


# ── Health & Status ───────────────────────────────────────────────────────────

@app.get("/health")
async def health_fast():
    """
    Fast health check — used by AI Engine 2's backend_client to verify
    the backend is reachable. Returns INSTANTLY with no outbound calls.
    """
    return {
        "status": "online",
        "system": "B.A.Y.M.A.X. v2.0",
        "backend_ip": os.getenv("TAILSCALE_IP", "100.108.247.7"),
    }


@app.get("/api/health")
async def health_check():
    """Full health check — probes AE2 status (may take up to 5s)."""
    ae2_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{_ae2_url()}/health")
            ae2_ok = r.status_code == 200 and r.json().get("status") in ("ok", "healthy", "degraded")
        _ae2_status_cache["online"] = ae2_ok
    except Exception:
        _ae2_status_cache["online"] = False

    return {
        "status": "online",
        "system": "B.A.Y.M.A.X. v2.0 Medical Intelligence",
        "backend_ip": os.getenv("TAILSCALE_IP", "100.108.247.7"),
        "ae2_url": _ae2_url(),
        "ae2_online": _ae2_status_cache["online"],
    }


# ── AI Engine Self-Registration ───────────────────────────────────────────────

@app.post("/ai-engine/register")
async def register_ai_engine(payload: AERegistrationPayload):
    """
    Called by AI Engine 2 on startup to register its Tailscale IP.
    This allows the backend to always proxy to the correct AE2 URL
    even if the IP changes.
    """
    if payload.engine == "ai_engine_2":
        _ae2_registry["url"] = payload.url.rstrip("/")
        logger.info(f"AI Engine 2 registered: {payload.url}")
        return {"status": "registered", "engine": payload.engine, "url": payload.url}
    return {"status": "ignored", "engine": payload.engine}


@app.get("/ai-engine/status")
async def ai_engine_status():
    """Return the currently registered AI Engine URLs."""
    return {
        "ae2_url": _ae2_url(),
        "ae2_configured_url": AE2_BASE_URL,
    }


# ── Core Chat ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    logger.info("Chat query", extra={"user_id": req.user_id, "query": req.query})

    mem = MemoryManager(db)
    mem.add_conversation_turn(req.user_id, "user", req.query)

    history_turns = mem.get_short_term_history(req.user_id)
    history_text  = "\n".join([f"{h['role']}: {h['message']}" for h in history_turns])

    pipeline_res = await reasoning_pipeline.execute_async(req.query, history_text)

    mem.add_conversation_turn(req.user_id, "assistant", pipeline_res.response)

    return ChatResponse(
        user_id=req.user_id,
        query=req.query,
        response=pipeline_res.response,
        confidence=pipeline_res.confidence.dict(),
        risk=pipeline_res.confidence.risk,
    )


# ── AI Engine 1 — Speech & Wake Word ──────────────────────────────────────────

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Speech-to-text running directly on AI Engine 1 (Faster-Whisper).
    """
    import tempfile, shutil
    from pathlib import Path

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await file.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = whisper_engine_ae1.transcribe_file(tmp_path)
        return {
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "duration_seconds": result.duration_s,
            "has_speech": not result.is_empty,
            "word_count": len(result.text.split()) if result.text else 0,
            "engine": "ai_engine_1",
        }
    except Exception as exc:
        logger.error(f"AI Engine 1 STT transcription error: {exc}")
        raise HTTPException(status_code=500, detail=f"AI Engine 1 STT failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/wakeword")
async def wakeword_detection(
    file: UploadFile = File(...),
    wake_word: str = Form("hey baymax")
):
    """
    Wake word detection running directly on AI Engine 1.
    Transcribes short audio chunks and checks if wake_word is present.
    """
    import tempfile
    from pathlib import Path

    target_phrase = (wake_word or "hey baymax").strip().lower()
    suffix = Path(file.filename or "chunk.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await file.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = whisper_engine_ae1.transcribe_file(tmp_path)
        transcript = (result.text or "").strip().lower()
        has_speech = not result.is_empty
        detected = (target_phrase in transcript) or ("baymax" in transcript)

        return {
            "has_speech": has_speech,
            "wake_word_detected": detected,
            "text": result.text,
            "confidence": result.confidence,
            "wake_word": target_phrase,
            "engine": "ai_engine_1",
        }
    except Exception as exc:
        logger.warning(f"AI Engine 1 Wake word error: {exc}")
        return {
            "has_speech": False,
            "wake_word_detected": False,
            "text": "",
            "confidence": 0.0,
            "error": str(exc),
            "engine": "ai_engine_1",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    TTS synthesis — tries AI Engine 2 (XTTS v2) first, falls back to local XTTS engine.
    Returns base64-encoded WAV audio.
    """
    import base64

    # 1. Try AE2 TTS proxy
    try:
        result = await _proxy_to_ae2(
            "/tts",
            json_body={
                "text": req.text,
                "voice": req.voice,
                "language": req.language,
                "stream": False,
                "format": "wav",
                "rvc_model_path": "k:\\PROJECTS\\BAYMAX\\Baymax_voice model\\Baymax.pth",
                "rvc_index_path": "k:\\PROJECTS\\BAYMAX\\Baymax_voice model\\added_IVF799_Flat_nprobe_1_Baymax_v2.index"
            },
        )
        return result
    except HTTPException:
        logger.warning("AE2 TTS unavailable, trying local XTTS engine...")

    # 2. Fall back to edge-tts (High quality Microsoft Edge voice)
    try:
        import edge_tts
        # en-US-ChristopherNeural is a very calm, deep, friendly male voice suitable for Baymax
        communicate = edge_tts.Communicate(req.text, "en-US-ChristopherNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        return {
            "success": True,
            "audio_base64": audio_b64,
            "duration_seconds": len(audio_data) / 16000, # Approximate
            "voice_used": "edge-tts-christopher",
        }
    except Exception as local_err:
        logger.warning(f"Edge-tts fallback failed: {local_err}")

    # 3. Return error — let frontend use browser TTS
    raise HTTPException(
        status_code=503,
        detail="TTS unavailable: AE2 offline and edge-tts failed. Browser TTS will be used.",
    )


# ── AI Engine 2 Proxy — Multimodal ───────────────────────────────────────────

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...), lang: str = Form("eng")):
    """
    Proxy OCR to AI Engine 2 (Tesseract + PyMuPDF).
    Falls back gracefully when AE2 is unavailable.
    """
    file_bytes = await file.read()
    try:
        return await _proxy_to_ae2(
            "/ocr",
            files={"file": (file.filename, file_bytes, file.content_type or "image/jpeg")},
            data={"lang": lang, "psm": "3", "oem": "3"},
        )
    except HTTPException:
        return JSONResponse(
            status_code=200,
            content={
                "available": False,
                "raw_text": "",
                "error": "AI Engine 2 is offline. OCR requires AE2 with Tesseract installed.",
                "document_type": "unknown",
                "extracted_fields": [],
                "overall_confidence": 0.0,
            },
        )


@app.post("/proxy/vision")
async def vision_endpoint(file: UploadFile = File(...)):
    """
    Proxy image vision analysis to AI Engine 2 with local AI Engine 1 fallback and local emotion overrides.
    Returns faces, emotions, person_detected, observations.
    """
    file_bytes = await file.read()
    
    # Run local AI Engine 1 emotion detection first to get accurate/updated emotions
    local_emotions = []
    local_observations = []
    try:
        engine = _get_emotion_engine()
        if engine is not None:
            result = engine.analyze_image_bytes(file_bytes)
            if result and result.current:
                score = result.current
                logger.info(f"Local emotion detected: {score.dominant_emotion} ({score.confidence:.2f})")
                local_emotions = [{"label": score.dominant_emotion, "confidence": score.confidence}]
                for k, v in score.normalized_scores.items():
                    if k != score.dominant_emotion:
                        local_emotions.append({"label": k, "confidence": v})
                local_observations = [
                    f"Face 0: apparent expression — {score.dominant_emotion} (confidence: {score.confidence:.2f})"
                ]
            else:
                logger.warning("Local EmotionEngine returned no result for uploaded frame")
        else:
            logger.warning("Local EmotionEngine singleton not available")
    except Exception as local_err:
        logger.warning(f"Local AI Engine 1 vision pre-analysis error: {local_err}")

    try:
        ae2_res = await _proxy_to_ae2(
            "/vision",
            files={"file": (file.filename, file_bytes, file.content_type or "image/jpeg")},
            data={
                "face_detection": "true",
                "emotion_context": "true",
                "roi_detection": "true",
                "person_detection": "true",
            },
        )
        if local_emotions and isinstance(ae2_res, dict):
            ae2_res["emotions"] = local_emotions
            if "observations" in ae2_res:
                obs = [o for o in ae2_res["observations"] if "apparent expression" not in o]
                obs.extend(local_observations)
                ae2_res["observations"] = obs
            ae2_res["engine"] = "ae2_with_local_emotions"
        return ae2_res
    except Exception as exc:
        logger.warning(f"AE2 vision analysis failed ({exc}), running local AI Engine 1 vision fallback...")
        if local_emotions:
            return {
                "person_detected": True,
                "faces": [{"box": [0, 0, 100, 100], "confidence": 0.9}],
                "emotions": local_emotions,
                "lighting_assessment": "normal",
                "observations": [
                    "Person: detected",
                    local_observations[0] if local_observations else "Face 0: apparent expression — neutral"
                ],
                "engine": "ai_engine_1_local_fallback"
            }

        # Basic image fallback response
        return {
            "person_detected": True,
            "faces": [{"box": [0, 0, 100, 100], "confidence": 0.85}],
            "emotions": [{"label": "neutral", "confidence": 0.70}],
            "lighting_assessment": "normal",
            "observations": [
                "Person detected in frame.",
                "Face 0: apparent expression — neutral (confidence: 0.70)",
                "Scene type: indoor"
            ],
            "engine": "ai_engine_1_safe_fallback"
        }


@app.post("/proxy/analyse")
async def analyse_endpoint(
    file: UploadFile = File(...),
    tasks: str = Form("ocr"),
):
    """
    Proxy combined parallel analysis to AI Engine 2's /analyse endpoint.
    ``tasks`` is a comma-separated list: e.g. "ocr,vision"
    """
    file_bytes = await file.read()
    return await _proxy_to_ae2(
        "/analyse",
        files={"file": (file.filename, file_bytes, file.content_type or "image/jpeg")},
        data={"tasks": tasks},
    )


@app.post("/proxy/document")
async def document_endpoint(file: UploadFile = File(...)):
    """
    Proxy structured document parsing to AI Engine 2 (PDF/image/text).
    """
    file_bytes = await file.read()
    content_type = file.content_type or "application/pdf"
    return await _proxy_to_ae2(
        "/document",
        files={"file": (file.filename, file_bytes, content_type)},
    )


@app.get("/proxy/ae2-health")
async def ae2_health_proxy():
    """Direct health probe of AI Engine 2. Never raises — always returns a status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{_ae2_url()}/health")
            return r.json()
    except Exception as exc:
        return {"status": "offline", "error": str(exc), "ae2_url": _ae2_url()}


@app.get("/proxy/ae2-status")
async def ae2_status_proxy():
    """Full status probe of AI Engine 2 (component readiness)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{_ae2_url()}/status")
            return r.json()
    except Exception as exc:
        return {"status": "offline", "error": str(exc)}


@app.get("/tts/voices")
async def list_voices():
    """List available TTS voice profiles from AI Engine 2."""
    return await _proxy_to_ae2("/tts/voices", method="GET")


# ── Conversation History ───────────────────────────────────────────────────────

@app.get("/history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
    mem = MemoryManager(db)
    return {"history": mem.get_short_term_history(user_id)}
