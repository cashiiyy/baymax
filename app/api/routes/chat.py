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
from app.llm.qwen_engine import QwenEngine
from app.llm.response_parser import ResponseParser
from app.memory.memory_manager import MemoryManager
from app.rag.pipeline import RAGPipeline
from app.tts.xtts_engine import XTTSEngine
from app.api.routes.avatar import avatar_ws_manager
from app.avatar.controller import AvatarController
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Full Pipeline"])

# ── Global Module Instances (Lazy Loaded) ──
rag_pipeline = RAGPipeline()
qwen_engine = QwenEngine()
response_parser = ResponseParser()
xtts_engine = XTTSEngine()
avatar_controller = AvatarController(avatar_ws_manager)


class ChatRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    text: str
    detected_emotion: str = "neutral"


class ChatResponse(BaseModel):
    response_text: str
    suggested_emotion: str
    is_emergency: bool
    latency_s: float
    # Base64 encoded WAV audio is optional here; usually streamed via WS
    audio_b64: Optional[str] = None


@router.post("/respond", response_model=ChatResponse)
async def chat_respond(req: ChatRequest):
    """
    Full pipeline execution for a text input.
    Use this endpoint if STT and Emotion detection are done client-side.
    """
    t_start = time.time()
    
    try:
        # 1. Initialize Memory Manager for this session
        memory = MemoryManager(user_id=req.user_id, session_id=req.session_id)
        await memory.initialize()
        
        # 2. Record User Turn
        await memory.record_turn(
            role="user", 
            content=req.text, 
            emotion=req.detected_emotion,
            store_as_memory=True  # Let vector memory decide if it's important
        )
        
        # 3. Build Context
        context_builder = ContextBuilder(
            rag_pipeline=rag_pipeline,
            memory_manager=memory
        )
        # We mock EmotionResult here based on the string provided by the client
        from app.emotion.deepface_engine import EmotionResult, EmotionScore
        mock_emotion = EmotionResult(
            current=EmotionScore(dominant_emotion=req.detected_emotion, scores={}),
            smoothed=EmotionScore(dominant_emotion=req.detected_emotion, scores={})
        )
        
        context = await context_builder.build(
            user_message=req.text,
            emotion_result=mock_emotion,
            user_id=req.user_id
        )
        
        # 4. Generate LLM Response
        llm_response = qwen_engine.generate(
            messages=context.chat_messages,
            stream=False
        )
        
        # 5. Parse Response
        parsed = response_parser.parse(llm_response.text)
        
        # 6. Record Assistant Turn
        await memory.record_turn(
            role="assistant",
            content=parsed.text,
            emotion=parsed.suggested_emotion
        )
        
        # 7. Generate Voice (TTS)
        speech = xtts_engine.synthesize(parsed.text)
        audio_bytes = speech.to_bytes(format="wav")
        
        # 8. Command Avatar
        await avatar_controller.speak(
            text=parsed.text,
            audio_bytes=audio_bytes,
            duration_s=speech.duration_s
        )
        
        # 9. Trigger expression based on parsed emotion
        await avatar_controller.set_emotion(parsed.avatar_emotion)
        
        elapsed = time.time() - t_start
        log.info("Full pipeline complete | latency={:.2f}s", elapsed)
        
        return ChatResponse(
            response_text=parsed.text,
            suggested_emotion=parsed.suggested_emotion,
            is_emergency=parsed.is_emergency,
            latency_s=round(elapsed, 2),
            audio_b64=None  # We sent it to avatar via WS, but could return here too
        )
        
    except Exception as exc:
        log.exception("Pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))
