"""
BAYMAX AI – Core Chat Pipeline Routes
=======================================
Wires together the entire system: 
Speech In → Emotion In → Memory → RAG → LLM → Voice Out → Avatar Out
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.context.builder import ContextBuilder
# BioMistral 7B on AI Engine 2 is the primary LLM provider
# Rollback: replace BioMistralEngine with QwenEngine and revert the import below
from app.llm.biomistral_engine import BioMistralEngine
# from app.llm.qwen_engine import QwenEngine  # ROLLBACK: uncomment this line
from app.llm.response_parser import ResponseParser
from app.memory.memory_manager import MemoryManager
from app.rag.pipeline import RAGPipeline
from app.tts.xtts_engine import XTTSEngine
from app.api.routes.avatar import avatar_ws_manager
from app.avatar.controller import AvatarController
from app.utils.ae2_client import ae2_client
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Full Pipeline"])

# ── Global Module Instances (Lazy Loaded) ──
rag_pipeline = RAGPipeline()
# BioMistral 7B on AI Engine 2 (primary)
bio_engine = BioMistralEngine()
# ROLLBACK: replace bio_engine with qwen_engine = QwenEngine()
response_parser = ResponseParser()
xtts_engine = XTTSEngine()
avatar_controller = AvatarController(avatar_ws_manager)


class ChatRequest(BaseModel):
    user_id: str = "1"
    session_id: Optional[str] = None
    text: Optional[str] = None
    query: Optional[str] = None
    detected_emotion: str = "neutral"

    @property
    def message_text(self) -> str:
        return (self.text or self.query or "").strip()


class ChatResponse(BaseModel):
    response_text: str
    suggested_emotion: str = "neutral"
    is_emergency: bool = False
    latency_s: float = 0.0
    # Base64 encoded WAV audio is optional here; usually streamed via WS
    audio_b64: Optional[str] = None


@router.post("/respond", response_model=ChatResponse)
@router.post("", response_model=ChatResponse)
async def chat_respond(req: ChatRequest):
    """
    Full pipeline execution for a text input.
    Use this endpoint if STT and Emotion detection are done client-side.
    """
    t_start = time.time()
    user_input = req.message_text
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing text or query in request.")
    
    try:
        # 1. Initialize Memory Manager for this session
        memory = MemoryManager(user_id=req.user_id, session_id=req.session_id)
        await memory.initialize()
        
        # 2. Record User Turn
        await memory.record_turn(
            role="user", 
            content=user_input, 
            emotion=req.detected_emotion,
            store_as_memory=True  # Let vector memory decide if it's important
        )
        
        # 3. Build Context
        context_builder = ContextBuilder(
            rag_pipeline=rag_pipeline,
            memory_manager=memory
        )
        from app.emotion.deepface_engine import EmotionResult, EmotionScore
        mock_emotion = EmotionResult(
            current=EmotionScore(dominant_emotion=req.detected_emotion, scores={}),
            smoothed=EmotionScore(dominant_emotion=req.detected_emotion, scores={})
        )
        
        context = await context_builder.build(
            user_message=user_input,
            emotion_result=mock_emotion,
            user_id=req.user_id
        )
        
        # 4. Generate LLM Response via BioMistral on AI Engine 2 (with automatic failover)
        try:
            llm_response = await bio_engine._generate_async(
                messages=context.chat_messages,
                max_tokens=1024,
            )
        except (ConnectionError, TimeoutError, Exception) as llm_err:
            log.warning("BioMistral inference unavailable ({}), attempting fallback via ProductionLLMEngine...", llm_err)
            try:
                from ai_engine_1.llm.llm_engine import ProductionLLMEngine
                engine1 = ProductionLLMEngine()
                sys_msg = next((m["content"] for m in context.chat_messages if m["role"] == "system"), None)
                usr_msg = next((m["content"] for m in reversed(context.chat_messages) if m["role"] == "user"), user_input)
                eng_res = await engine1.generate_async(prompt=usr_msg, system_prompt=sys_msg)
                from app.llm.biomistral_engine import LLMResponse
                llm_response = LLMResponse(
                    text=eng_res.text,
                    input_tokens=0,
                    output_tokens=eng_res.tokens_used,
                    elapsed_s=eng_res.latency_ms / 1000.0,
                    provider=eng_res.model_used,
                )
            except Exception as fb_err:
                log.error("All LLM fallbacks failed: {}", fb_err)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"BioMistral inference service unavailable: {llm_err}. "
                        "AI Engine 2 may be offline or LM Studio is not running."
                    )
                )
        
        # 5. Parse Response
        parsed = response_parser.parse(llm_response.text)
        
        # 6. Record Assistant Turn
        await memory.record_turn(
            role="assistant",
            content=parsed.text,
            emotion=parsed.suggested_emotion
        )
        
        # 7. Generate Voice (TTS) — Decoupled & Isolated: Voicebox -> AE2 -> Local XTTS
        audio_b64: str | None = None
        audio_bytes = b""
        duration_s = 3.0

        try:
            from config import settings
            if getattr(settings, "VOICEBOX_ENABLED", True):
                from app.tts.voicebox_engine import voicebox_engine
                vb_res = await voicebox_engine.generate_speech(parsed.text)
                if vb_res.get("success") and vb_res.get("audio_base64"):
                    audio_b64 = vb_res["audio_base64"]
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)
                    duration_s = vb_res.get("duration_seconds", 3.0)
                    log.info("TTS via Voicebox ({}) | duration={:.2f}s", vb_res.get("voice_used"), duration_s)
        except Exception as vb_err:
            log.warning("Voicebox TTS unavailable ({}), trying AE2 / XTTS fallback...", vb_err)

        if not audio_b64:
            try:
                ae2_tts = await ae2_client.tts(parsed.text)
                if ae2_tts and ae2_tts.get("audio_base64"):
                    audio_b64 = ae2_tts["audio_base64"]
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)
                    duration_s = ae2_tts.get("duration_seconds", 3.0)
                    log.info("TTS via AI Engine 2 | duration={:.2f}s", duration_s)
            except Exception as ae2_err:
                log.warning("AE2 TTS unavailable ({}), trying local XTTS.", ae2_err)
                try:
                    speech = xtts_engine.synthesize(parsed.text)
                    audio_bytes = speech.to_bytes(format="wav")
                    duration_s = speech.duration_s
                    import base64
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                except Exception:
                    log.info("Local XTTS unavailable. Returning text without audio.")
                    audio_bytes = b""
                    duration_s = 3.0
                    audio_b64 = None
        
        # 8. Command Avatar (non-blocking notification)
        try:
            await avatar_controller.speak(
                text=parsed.text,
                audio_bytes=audio_bytes,
                duration_s=duration_s
            )
            # 9. Trigger expression based on parsed emotion
            await avatar_controller.set_emotion(parsed.avatar_emotion)
        except Exception as av_err:
            log.warning("Avatar controller notification failed: {}", av_err)
        
        elapsed = time.time() - t_start
        log.info("Full pipeline complete | latency={:.2f}s", elapsed)
        
        return ChatResponse(
            response_text=parsed.text,
            suggested_emotion=parsed.suggested_emotion,
            is_emergency=parsed.is_emergency,
            latency_s=round(elapsed, 2),
            audio_b64=audio_b64,
        )
        
    except Exception as exc:
        log.exception("Pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))
