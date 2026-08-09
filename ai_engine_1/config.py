import os
from pathlib import Path
from pydantic import BaseModel, Field

# ── Load .env from project root (K:\PROJECTS\BAYMAX\.env) ─────────────────────
try:
    from dotenv import load_dotenv
    # Walk up from this file's location to find the .env
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars set externally

class EngineConfig(BaseModel):
    """Configuration settings for AI Engine 1 Medical Intelligence Engine."""
    # LLM Settings (OmniRoute API Local Gateway proxy)
    omniroute_api_key: str = Field(default_factory=lambda: os.getenv("OMNIROUTE_API_KEY", ""))
    omniroute_base_url: str = Field(default_factory=lambda: os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1"))
    primary_llm_model: str = Field(default_factory=lambda: os.getenv("PRIMARY_LLM_MODEL", "auto/best-free"))
    
    # Fallback LLM Settings (Local Ollama Qwen 2.5 7B or 1.5B fallback)
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    fallback_llm_model: str = Field(default_factory=lambda: os.getenv("FALLBACK_LLM_MODEL", "qwen2.5:7b"))
    
    # Embedding Settings
    embedding_model_name: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
    embedding_dim: int = 384
    embedding_cache_size: int = 2048
    
    # RAG Settings
    vector_top_k: int = 5
    similarity_threshold: float = 0.45
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Local Qwen 2.5 Settings (4-bit quantized, fits 8GB VRAM)
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "omniroute"))
    qwen_model: str = Field(default_factory=lambda: os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"))
    qwen_max_new_tokens: int = Field(default_factory=lambda: int(os.getenv("QWEN_MAX_NEW_TOKENS", "512")))
    qwen_temperature: float = Field(default_factory=lambda: float(os.getenv("QWEN_TEMPERATURE", "0.3")))
    qwen_top_p: float = Field(default_factory=lambda: float(os.getenv("QWEN_TOP_P", "0.9")))
    qwen_repetition_penalty: float = Field(default_factory=lambda: float(os.getenv("QWEN_REPETITION_PENALTY", "1.1")))

    # Safety & Confidence Settings
    min_confidence_threshold: float = 0.65
    high_confidence_threshold: float = 0.85
    max_context_tokens: int = 4096
    
    # Service Settings
    engine1_host: str = Field(default_factory=lambda: os.getenv("ENGINE1_HOST", "0.0.0.0"))
    engine1_port: int = Field(default_factory=lambda: int(os.getenv("ENGINE1_PORT", "8001")))

settings = EngineConfig()
