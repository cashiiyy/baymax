"""
BAYMAX AI – Main FastAPI Application Entry Point
=================================================
Initializes the FastAPI app, registers middleware, and mounts all routers.
Run this file directly to start the uvicorn server.

Usage:
    python main.py
"""

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from config import settings
from app.api.middleware import add_middleware
from app.api.routes import health, stt, emotion, chat, memory, medical, avatar, multimodal
from app.utils.logger import setup_logging, get_logger

# ── Setup Logging ──
setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    log.info("Starting BAYMAX AI Server v{}", settings.APP_VERSION)
    
    # Pre-flight checks
    from app.utils.gpu import gpu_info
    gpu_info()  # Will log GPU status
    
    # Initialize RAG Pipeline checks
    from app.rag.pipeline import RAGPipeline
    try:
        rag = RAGPipeline()
        status = rag.status()
        if not status.is_built:
            log.warning("RAG vector database is empty. You should run a data ingestion build.")
        else:
            log.info("RAG vector database ready | counts: {}", status.collection_counts)
    except Exception as exc:
        log.error("Failed to check RAG status: {}", exc)

    yield  # Server runs here
    
    # Shutdown events
    log.info("Shutting down BAYMAX AI Server")
    from app.llm.qwen_engine import QwenEngine
    from app.tts.xtts_engine import XTTSEngine
    from app.utils.ae2_client import ae2_client
    QwenEngine().unload()
    XTTSEngine().unload()
    await ae2_client.close()  # Close AE2 httpx connection pool


# ── Create FastAPI App ──
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal Healthcare Assistant AI",
    lifespan=lifespan,
)

# ── Register Middleware ──
add_middleware(app)

# ── Mount Routers ──
app.include_router(health.router)
app.include_router(stt.router)
app.include_router(emotion.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(medical.router)
app.include_router(avatar.router)
app.include_router(multimodal.router)


if __name__ == "__main__":
    log.info("Starting Uvicorn server on {}:{}", settings.API_HOST, settings.API_PORT)
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="warning",  # Let loguru handle our logs
    )
