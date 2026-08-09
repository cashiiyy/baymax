"""
BAYMAX AI – RAG Pipeline Orchestrator
=======================================
Coordinates the full RAG ingestion pipeline:
  DatasetPreprocessor → DocumentChunker → RAGEmbedder → MedicalVectorStore

Also provides the top-level search interface used by other modules.

Usage:
    from app.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.build()                    # One-time dataset ingestion
    result = pipeline.search("fever")   # Runtime retrieval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.datasets.preprocessor import DatasetPreprocessor
from app.rag.chunker import DocumentChunk, DocumentChunker
from app.rag.embedder import RAGEmbedder
from app.rag.retriever import MedicalRetriever, MedicalRetrievalResult
from app.rag.vector_store import MedicalVectorStore
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PipelineStatus:
    """Reports the current state of the RAG pipeline."""
    is_built: bool = False
    collection_counts: Dict[str, int] = None  # type: ignore[assignment]
    embedding_dim: int = 0
    processed_datasets: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.collection_counts is None:
            self.collection_counts = {}
        if self.processed_datasets is None:
            self.processed_datasets = []


class RAGPipeline:
    """
    End-to-end RAG pipeline for medical knowledge ingestion and retrieval.

    Lifecycle:
        1. `build()`:  Preprocess datasets → chunk → embed → store in ChromaDB
        2. `search()`: Embed query → similarity search → return ranked results

    Attributes:
        preprocessor:  Dataset preprocessor instance.
        chunker:       Document chunker instance.
        embedder:      Text embedder instance.
        vector_store:  ChromaDB vector store instance.
        retriever:     Semantic retriever instance.
    """

    # Maps processed JSON filenames → ChromaDB collection names
    DATASET_COLLECTION_MAP: Dict[str, str] = {
        "diseases.json":      "disease_knowledge",
        "symptoms.json":      "symptom_knowledge",
        "medicines.json":     "medicine_knowledge",
        "firstaid.json":      "firstaid_knowledge",
        "general_health.json": "general_health",
    }

    def __init__(
        self,
        preprocessor: Optional[DatasetPreprocessor] = None,
        chunker: Optional[DocumentChunker] = None,
        embedder: Optional[RAGEmbedder] = None,
        vector_store: Optional[MedicalVectorStore] = None,
    ) -> None:
        from config import settings

        self._settings = settings
        self.preprocessor = preprocessor or DatasetPreprocessor()
        self.chunker = chunker or DocumentChunker()
        self.embedder = embedder or RAGEmbedder()
        self.vector_store = vector_store or MedicalVectorStore()
        self.retriever = MedicalRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        log.info("RAGPipeline initialized")

    # ── Build Phase ───────────────────────────────────────────────────────────

    def build(self, force_rebuild: bool = False) -> PipelineStatus:
        """
        Run the full ingestion pipeline.

        Steps:
            1. Check if collections already exist (skip if not force_rebuild)
            2. Run DatasetPreprocessor on raw datasets
            3. Load processed JSON files
            4. Chunk, embed, and store in ChromaDB

        Args:
            force_rebuild: If True, re-index even if collections are already populated.

        Returns:
            PipelineStatus with collection counts.
        """
        log.info("RAG pipeline build started | force_rebuild={}", force_rebuild)

        # Check if already built
        if not force_rebuild and self._is_already_built():
            log.info("RAG pipeline already built. Use force_rebuild=True to re-index.")
            return self.status()

        # Step 1: Preprocess raw datasets
        log.info("Step 1/3: Preprocessing datasets...")
        self.preprocessor.process_all()

        # Step 2: Load processed files
        log.info("Step 2/3: Loading processed datasets...")
        processed_dir = self._settings.DATASETS_PROCESSED_DIR
        total_chunks = 0

        for filename, collection_name in self.DATASET_COLLECTION_MAP.items():
            file_path = processed_dir / filename
            if not file_path.exists():
                log.warning("Processed file not found: {}", filename)
                continue

            try:
                records = self._load_processed_file(file_path)
                if not records:
                    continue

                # Step 3: Chunk + embed + store
                chunks = self._ingest_records(records, collection_name, filename)
                total_chunks += chunks
                log.info(
                    "Ingested {} | {} records → {} chunks",
                    filename,
                    len(records),
                    chunks,
                )
            except Exception as exc:
                log.error("Failed to ingest {}: {}", filename, exc)

        log.info(
            "RAG pipeline build complete | total_chunks={}",
            total_chunks,
        )
        return self.status()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> MedicalRetrievalResult:
        """
        Retrieve relevant medical information for a query.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            MedicalRetrievalResult.
        """
        return self.retriever.retrieve(query, top_k=top_k)

    def status(self) -> PipelineStatus:
        """Return the current pipeline status with collection document counts."""
        from config import settings

        counts = {}
        collections = [
            settings.CHROMA_COLLECTION_DISEASE,
            settings.CHROMA_COLLECTION_SYMPTOM,
            settings.CHROMA_COLLECTION_MEDICINE,
            settings.CHROMA_COLLECTION_FIRSTAID,
            settings.CHROMA_COLLECTION_GENERAL,
        ]
        for name in collections:
            try:
                counts[name] = self.vector_store.collection_count(name)
            except Exception:
                counts[name] = 0

        is_built = any(c > 0 for c in counts.values())
        return PipelineStatus(
            is_built=is_built,
            collection_counts=counts,
            embedding_dim=self.embedder.embedding_dim if is_built else 0,
            processed_datasets=list(self.DATASET_COLLECTION_MAP.keys()),
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _is_already_built(self) -> bool:
        """Check if any collections already have documents."""
        from config import settings

        for name in [
            settings.CHROMA_COLLECTION_DISEASE,
            settings.CHROMA_COLLECTION_SYMPTOM,
        ]:
            if self.vector_store.collection_exists(name):
                return True
        return False

    def _load_processed_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load a processed JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _ingest_records(
        self,
        records: List[Dict[str, Any]],
        collection_name: str,
        source: str,
    ) -> int:
        """Chunk → embed → store a list of records."""
        # Build text representations from records
        texts = []
        metadatas = []

        for idx, record in enumerate(records):
            # Use to_text() if available (JSON was serialized from dataclass)
            # Otherwise join all string fields
            text = self._record_to_text(record)
            if not text.strip():
                continue
            texts.append(text)
            rec_id = record.get("disease_id") or record.get("symptom_id") \
                     or record.get("medicine_id") or record.get("aid_id") \
                     or record.get("record_id") or record.get("") or f"idx_{idx}"
            metadatas.append({
                "source": source,
                "collection": collection_name,
                "record_id": str(rec_id),
            })

        if not texts:
            return 0

        # Chunk
        all_chunks: List[DocumentChunk] = []
        all_metas: List[dict] = []
        for text, meta in zip(texts, metadatas):
            chunks = self.chunker.chunk_text(text, meta)
            all_chunks.extend(chunks)
            all_metas.extend([c.metadata for c in chunks])

        chunk_texts = [c.text for c in all_chunks]
        chunk_ids = [c.chunk_id for c in all_chunks]

        # Embed in batches
        log.info(
            "Embedding {} chunks for collection '{}'...",
            len(chunk_texts),
            collection_name,
        )
        embeddings = self.embedder.embed(chunk_texts, show_progress=True)

        # Store
        self.vector_store.add_documents(
            collection_name=collection_name,
            texts=chunk_texts,
            embeddings=embeddings,
            metadatas=all_metas,
            ids=chunk_ids,
        )
        return len(chunk_texts)

    @staticmethod
    def _record_to_text(record: Dict[str, Any]) -> str:
        """Convert a record dict to a flat text string for embedding."""
        parts = []
        for key, value in record.items():
            if key.endswith("_id") or not value:
                continue
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, str):
                parts.append(f"{key}: {value}")
            elif isinstance(value, bool):
                parts.append(f"{key}: {'yes' if value else 'no'}")
        return " | ".join(parts)
