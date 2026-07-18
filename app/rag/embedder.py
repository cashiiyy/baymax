"""
BAYMAX AI – RAG Embedder
=========================
Generates dense vector embeddings for text chunks using
SentenceTransformers. Supports batch processing and result caching.

Usage:
    from app.rag.embedder import RAGEmbedder
    embedder = RAGEmbedder()
    vectors = embedder.embed(["flu symptoms", "headache treatment"])
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


class RAGEmbedder:
    """
    Text embedder using SentenceTransformers.

    Features:
        - Lazy model loading (first call triggers download/load)
        - Batch processing with configurable batch size
        - Disk-based embedding cache (JSON) to avoid re-computing embeddings
        - GPU acceleration if available

    Attributes:
        model_name:  HuggingFace model identifier.
        device:      Compute device ('cuda' or 'cpu').
        batch_size:  Number of texts to encode per forward pass.
        cache_path:  Optional path for persistent embedding cache.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_path: Optional[Path] = None,
    ) -> None:
        from config import settings
        from app.utils.gpu import get_device

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or get_device()
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._model = None  # Lazy loading
        self._cache: dict[str, List[float]] = {}

        if cache_path:
            self._load_cache()

        log.info(
            "RAGEmbedder configured | model={} device={} batch={}",
            self.model_name,
            self.device,
            self.batch_size,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts:         List of text strings to embed.
            show_progress: If True, show a tqdm progress bar.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        self._ensure_model_loaded()
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts in batches
        if uncached_texts:
            log.debug(
                "Embedding {}/{} texts (cache hit: {})",
                len(uncached_texts),
                len(texts),
                len(texts) - len(uncached_texts),
            )
            embeddings = self._model.encode(  # type: ignore[union-attr]
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )
            for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
                emb_list = emb.tolist()
                results[idx] = emb_list
                self._cache[self._cache_key(text)] = emb_list

            if self.cache_path:
                self._save_cache()

        return results  # type: ignore[return-value]

    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Input string.

        Returns:
            Embedding vector.
        """
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        self._ensure_model_loaded()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is not None:
            return
        log.info("Loading embedding model: {}", self.model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        log.info(
            "Embedding model loaded | dim={}", self.embedding_dim
        )

    @staticmethod
    def _cache_key(text: str) -> str:
        """Compute a short hash key for a text string."""
        return hashlib.sha1(text.encode()).hexdigest()

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
                log.info(
                    "Embedding cache loaded | {} entries", len(self._cache)
                )
            except Exception as exc:
                log.warning("Failed to load embedding cache: {}", exc)
                self._cache = {}

    def _save_cache(self) -> None:
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))
        log.debug("Embedding cache saved | {} entries", len(self._cache))
