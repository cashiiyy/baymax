"""
BAYMAX AI – ChromaDB Vector Store
===================================
Persistent vector store using ChromaDB. One collection per dataset type.
Handles document ingestion, upsert, and semantic similarity search.

Usage:
    from app.rag.vector_store import MedicalVectorStore
    store = MedicalVectorStore()
    store.add_documents(collection="disease_knowledge", chunks=[...])
    results = store.query(collection="disease_knowledge", query_embedding=[...], k=5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    """
    A single result from a vector similarity search.

    Attributes:
        doc_id:     Document/chunk ID.
        text:       The matched text chunk.
        metadata:   Associated metadata dict.
        score:      Similarity score (higher = more similar, 0–1 for cosine).
        collection: Source collection name.
    """
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    collection: str = ""


class MedicalVectorStore:
    """
    ChromaDB-backed persistent vector store for the medical knowledge base.

    Collections:
        - disease_knowledge
        - symptom_knowledge
        - medicine_knowledge
        - firstaid_knowledge
        - general_health
        - user_episodic_memory  (used by MemoryEngine)

    Attributes:
        persist_dir: Path to ChromaDB persistent storage.
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        from config import settings
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        dir_path = persist_dir or str(settings.CHROMA_DB_DIR)

        self._client = chromadb.PersistentClient(
            path=dir_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: Dict[str, Any] = {}
        log.info("MedicalVectorStore initialized | persist_dir={}", dir_path)

    # ── Collection Management ─────────────────────────────────────────────────

    def get_or_create_collection(self, name: str) -> Any:
        """
        Get an existing collection or create it if it doesn't exist.

        Args:
            name: Collection name.

        Returns:
            ChromaDB Collection object.
        """
        if name not in self._collections:
            collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},  # Use cosine distance
            )
            self._collections[name] = collection
            log.info(
                "Collection ready | name={} | docs={}",
                name,
                collection.count(),
            )
        return self._collections[name]

    def collection_count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        col = self.get_or_create_collection(collection_name)
        return col.count()

    def collection_exists(self, collection_name: str) -> bool:
        """Return True if a collection has any documents."""
        return self.collection_count(collection_name) > 0

    # ── Document Ingestion ────────────────────────────────────────────────────

    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
    ) -> int:
        """
        Add documents (with pre-computed embeddings) to a collection.

        Args:
            collection_name: Target collection name.
            texts:           List of text strings.
            embeddings:      Corresponding embedding vectors.
            metadatas:       Optional list of metadata dicts.
            ids:             Optional list of document IDs (auto-generated if None).
            batch_size:      Chroma upsert batch size.

        Returns:
            Number of documents added.
        """
        if not texts:
            log.warning("add_documents called with empty texts list")
            return 0

        collection = self.get_or_create_collection(collection_name)
        metas = metadatas or [{} for _ in texts]
        doc_ids = ids or [self._generate_id(t) for t in texts]

        # Ensure metadata values are all strings/ints/floats/bools (ChromaDB requirement)
        clean_metas = [self._sanitize_metadata(m) for m in metas]

        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_slice = slice(i, i + batch_size)
            collection.upsert(
                documents=texts[batch_slice],
                embeddings=embeddings[batch_slice],
                metadatas=clean_metas[batch_slice],
                ids=doc_ids[batch_slice],
            )
            total_added += len(texts[batch_slice])

        log.info(
            "Documents upserted | collection={} | count={}",
            collection_name,
            total_added,
        )
        return total_added

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform a vector similarity search in a collection.

        Args:
            collection_name:  Target collection.
            query_embedding:  Query vector (same dim as stored embeddings).
            k:                Number of results to return.
            where:            Optional ChromaDB metadata filter.

        Returns:
            List of SearchResult objects, ordered by similarity (desc).
        """
        collection = self.get_or_create_collection(collection_name)

        if collection.count() == 0:
            log.warning("Querying empty collection: {}", collection_name)
            return []

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = collection.query(**query_params)

        search_results: List[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist, doc_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                # Convert cosine distance → similarity score (1 = identical)
                score = max(0.0, 1.0 - dist)
                search_results.append(
                    SearchResult(
                        doc_id=doc_id,
                        text=doc,
                        metadata=meta or {},
                        score=round(score, 4),
                        collection=collection_name,
                    )
                )

        log.debug(
            "Query completed | collection={} | results={}",
            collection_name,
            len(search_results),
        )
        return search_results

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection (use with caution)."""
        self._client.delete_collection(collection_name)
        self._collections.pop(collection_name, None)
        log.warning("Collection deleted: {}", collection_name)

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _generate_id(text: str) -> str:
        import hashlib
        return hashlib.sha1(text.encode()).hexdigest()[:20]

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are ChromaDB-compatible types."""
        clean: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean
