"""
BAYMAX AI – Central Configuration
==================================
All system-wide configuration is managed here via environment variables
or a .env file. Never hardcode values in module files.

Usage:
    from config import settings
    print(settings.WHISPER_MODEL)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Base Paths ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent


class BaymaxSettings(BaseSettings):
    """
    Central settings object for the entire BAYMAX AI system.
    Values are loaded from environment variables or a `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "BAYMAX AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_KEY: str = ""  # Leave empty to disable auth

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    DATASETS_RAW_DIR: Path = BASE_DIR / "app" / "datasets" / "raw"
    DATASETS_PROCESSED_DIR: Path = BASE_DIR / "app" / "datasets" / "processed"
    CHROMA_DB_DIR: Path = BASE_DIR / "data" / "chroma_db"
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "baymax.db"
    BAYMAX_VOICE_REF: Path = BASE_DIR / "models" / "baymax_voice_ref.wav"
    LOG_DIR: Path = BASE_DIR / "logs"

    # ── Whisper STT ───────────────────────────────────────────────────────────
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "medium"
    WHISPER_LANGUAGE: str = "en"
    WHISPER_COMPUTE_TYPE: Literal["int8", "float16", "float32"] = "int8"
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_SIZE: int = 1024
    SILENCE_THRESHOLD_DB: float = -40.0
    VAD_AGGRESSIVENESS: int = 2  # 0–3

    # ── Emotion Detection ─────────────────────────────────────────────────────
    DEEPFACE_MODEL: str = "Emotion"
    DEEPFACE_BACKEND: Literal[
        "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"
    ] = "mediapipe"
    WEBCAM_INDEX: int = 0
    WEBCAM_FPS: int = 15
    EMOTION_SMOOTHING_FRAMES: int = 5  # Temporal smoothing window

    # ── RAG / Embeddings ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5
    CHROMA_COLLECTION_DISEASE: str = "disease_knowledge"
    CHROMA_COLLECTION_SYMPTOM: str = "symptom_knowledge"
    CHROMA_COLLECTION_MEDICINE: str = "medicine_knowledge"
    CHROMA_COLLECTION_FIRSTAID: str = "firstaid_knowledge"
    CHROMA_COLLECTION_GENERAL: str = "general_health"

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_MODEL_ID: str = "Qwen/Qwen3-8B-Instruct"
    LLM_LOAD_IN_4BIT: bool = False  # RTX 5050 8GB → use 8-bit instead
    LLM_LOAD_IN_8BIT: bool = True   # 8-bit quantization for 8GB VRAM
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_REPETITION_PENALTY: float = 1.1
    LLM_DEVICE_MAP: str = "auto"
    LLM_TORCH_DTYPE: str = "float16"  # RTX 5050 supports FP16

    # ── TTS ───────────────────────────────────────────────────────────────────
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    TTS_LANGUAGE: str = "en"
    TTS_SAMPLE_RATE: int = 24000
    TTS_STREAMING: bool = True

    # ── Memory ───────────────────────────────────────────────────────────────
    SHORT_TERM_WINDOW: int = 20       # Max messages in short-term memory
    VECTOR_MEMORY_TOP_K: int = 3      # Retrieved episodic memories per query
    CHROMA_COLLECTION_MEMORY: str = "user_episodic_memory"

    # ── Avatar ────────────────────────────────────────────────────────────────
    AVATAR_WS_HOST: str = "0.0.0.0"
    AVATAR_WS_PORT: int = 8001

    @field_validator("MODELS_DIR", "DATA_DIR", "CHROMA_DB_DIR", "LOG_DIR", mode="before")
    @classmethod
    def create_directory(cls, v: Path | str) -> Path:
        """Ensure all configured directories exist on startup."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def auth_enabled(self) -> bool:
        """Returns True if API key authentication is configured."""
        return bool(self.API_KEY)

    def device(self) -> str:
        """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ── Global Singleton ──────────────────────────────────────────────────────────
settings = BaymaxSettings()
