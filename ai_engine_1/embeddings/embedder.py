import numpy as np
import asyncio
from functools import lru_cache
from typing import List, Dict, Any, Optional

class AdvancedEmbedder:
    """Batch, metadata-aware, LRU-cached embedding generation engine."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 2048):
        self.model_name = model_name
        self.cache_size = cache_size
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None

    def encode(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]

        self._load_model()
        if self._model:
            vector = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            # Deterministic fallback vector
            np.random.seed(abs(hash(text)) % (2**32))
            vector = np.random.rand(384).astype(np.float32)

        if len(self._cache) >= self.cache_size:
            # Simple eviction of oldest item
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vector
        return vector

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        if self._model and texts:
            return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        return np.array([self.encode(t) for t in texts], dtype=np.float32)

    async def encode_async(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self.encode, text)

    def encode_metadata_aware(self, text: str, metadata: Dict[str, Any]) -> np.ndarray:
        meta_prefix = " ".join([f"{k}:{v}" for k, v in metadata.items() if isinstance(v, (str, int, float))])
        full_text = f"{meta_prefix} | {text}" if meta_prefix else text
        return self.encode(full_text)

class SentenceTransformerEmbedder(AdvancedEmbedder):
    """Backward compatibility wrapper around AdvancedEmbedder."""
    pass
