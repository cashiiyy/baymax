import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List

from backend.database.db import get_db, engine, Base
from backend.database.models import User, Conversation
from backend.memory.memory_manager import MemoryManager
from backend.rag.rag_engine import RAGEngine
from backend.logging.logger import get_logger

from ai_engine_1.llm.qwen import QwenLLM
from ai_engine_1.embeddings.embedder import SentenceTransformerEmbedder
from ai_engine_2.whisper.transcribe import WhisperTranscriber
from ai_engine_2.xtts.synthesize import XTTSSynthesizer
from ai_engine_2.ocr.ocr import MedicalOCR

logger = get_logger("baymax-backend")

# Ensure tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="B.A.Y.M.A.X. v1 API & Web Interface",
    description="Local Multimodal AI Health Assistant API Gateway and Web Application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory setup
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize engines
embedder = SentenceTransformerEmbedder()
rag_engine = RAGEngine()
llm = QwenLLM()
transcriber = WhisperTranscriber()
synthesizer = XTTSSynthesizer()
ocr = MedicalOCR()

class ChatRequest(BaseModel):
    user_id: int = 1
    query: str

class ChatResponse(BaseModel):
    user_id: int
    query: str
    response: str

@app.get("/")
@app.get("/ui")
def serve_web_interface():
    """Serves the Red & White B.A.Y.M.A.X. Web Interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "B.A.Y.M.A.X. v1 Backend Online. Web interface files missing."}

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "B.A.Y.M.A.X. v1",
        "tailscale_ip": os.getenv("TAILSCALE_IP", "100.89.251.123")
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    logger.info("Received chat query", extra={"user_id": req.user_id, "query": req.query})
    
    mem = MemoryManager(db)
    mem.add_conversation_turn(req.user_id, "user", req.query)
    
    history = mem.get_short_term_history(req.user_id)
    prompt = rag_engine.format_prompt(req.query, [], history)
    
    raw_answer = llm.generate(prompt)
    safe_answer = rag_engine.enforce_safety(raw_answer)
    
    mem.add_conversation_turn(req.user_id, "assistant", safe_answer)
    return ChatResponse(user_id=req.user_id, query=req.query, response=safe_answer)

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        transcript = transcriber.transcribe(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return {"transcript": transcript}

@app.post("/tts")
def text_to_speech(text: str = Form(...), speaker_wav: Optional[str] = Form(None)):
    out_path = "output_speech.wav"
    synthesizer.synthesize_to_file(text, out_path, speaker_wav=speaker_wav)
    return FileResponse(out_path, media_type="audio/wav", filename="baymax_speech.wav")

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_ocr_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        extracted_text = ocr.extract_text(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return {"extracted_text": extracted_text}

@app.get("/history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
    mem = MemoryManager(db)
    return {"history": mem.get_short_term_history(user_id)}
