"""
BAYMAX AI – Vector Memory (Episodic Memory)
============================================
Stores and retrieves user episodic memories using ChromaDB.
Memories are conversation summaries automatically saved and retrieved
to provide long-term context for personalized responses.

Usage:
    from app.memory.vector_memory import VectorMemory
    vm = VectorMemory(user_id="user1")
    vm.store_memory("User mentioned they have a peanut allergy")
    memories = vm.retrieve_relevant("what are my allergies?")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from app.rag.embedder import RAGEmbedder
from app.rag.vector_store import MedicalVectorStore, SearchResult
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class EpisodicMemory:
    """
    A single stored episodic memory entry.

    Attributes:
        memory_id:   Unique identifier.
        content:     Memory text content.
        user_id:     Owner user ID.
        session_id:  Session where this memory was formed.
        timestamp:   Unix timestamp.
        importance:  Importance score (0–1) for retrieval weighting.
    """
    memory_id: str
    content: str
    user_id: str
    session_id: str
    timestamp: float
    importance: float = 0.5


class VectorMemory:
    """
    Long-term episodic memory using ChromaDB vector store.

    Each user has a dedicated memory namespace within the shared collection.
    Memories are automatically embedded and retrieved by semantic similarity.

    Attributes:
        user_id:      Owner of these memories.
        embedder:     RAGEmbedder for encoding memory content.
        vector_store: ChromaDB vector store.
        top_k:        Number of memories to retrieve per query.
    """

    def __init__(
        self,
        user_id: str,
        embedder: Optional[RAGEmbedder] = None,
        vector_store: Optional[MedicalVectorStore] = None,
        top_k: Optional[int] = None,
    ) -> None:
        from config import settings

        self.user_id = user_id
        self.embedder = embedder or RAGEmbedder()
        self.vector_store = vector_store or MedicalVectorStore()
        self.top_k = top_k or settings.VECTOR_MEMORY_TOP_K
        self._collection = settings.CHROMA_COLLECTION_MEMORY
        log.info(
            "VectorMemory initialized | user={} collection={}",
            user_id,
            self._collection,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        session_id: str = "default",
        importance: float = 0.5,
    ) -> str:
        """
        Store a new episodic memory for this user.

        Args:
            content:    Memory text to store.
            session_id: Source session ID.
            importance: Importance weight (0–1).

        Returns:
            Memory ID of the stored entry.
        """
        if not content.strip():
            return ""

        import hashlib

        memory_id = hashlib.sha1(
            f"{self.user_id}:{content}:{time.time()}".encode()
        ).hexdigest()[:16]

        embedding = self.embedder.embed_single(content)

        self.vector_store.add_documents(
            collection_name=self._collection,
            texts=[content],
            embeddings=[embedding],
            metadatas=[{
                "user_id": self.user_id,
                "session_id": session_id,
                "timestamp": str(time.time()),
                "importance": str(importance),
            }],
            ids=[memory_id],
        )

        log.info(
            "Memory stored | user={} id={} content='{}'",
            self.user_id,
            memory_id,
            content[:60],
        )
        return memory_id

    def retrieve_relevant(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Retrieve memories relevant to a query.

        Args:
            query:  Query string to find similar memories.
            top_k:  Number of memories to return.

        Returns:
            List of SearchResult, sorted by relevance.
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_single(query)

        results = self.vector_store.query(
            collection_name=self._collection,
            query_embedding=query_embedding,
            k=k,
            where={"user_id": self.user_id},  # User-scoped retrieval
        )

        log.debug(
            "Retrieved {} memories for user={} query='{}'",
            len(results),
            self.user_id,
            query[:60],
        )
        return results

    def retrieve_context_string(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve memories and format as context string for the LLM.

        Args:
            query:  Query string.
            top_k:  Number of memories.

        Returns:
            Formatted memory context string.
        """
        memories = self.retrieve_relevant(query, top_k=top_k)
        if not memories:
            return "No relevant previous memories found."

        lines = ["=== RETRIEVED USER MEMORIES ==="]
        for i, mem in enumerate(memories, 1):
            lines.append(f"[{i}] (relevance={mem.score:.2f}) {mem.text}")
        return "\n".join(lines)

    def store_conversation_summary(
        self,
        summary: str,
        session_id: str,
    ) -> str:
        """
        Store an auto-generated summary of a conversation session.

        Args:
            summary:    Text summary of the session.
            session_id: Session identifier.

        Returns:
            Memory ID.
        """
        content = f"[Session {session_id[:8]}] {summary}"
        return self.store_memory(
            content=content,
            session_id=session_id,
            importance=0.7,
        )

    def get_all_memories(self) -> List[SearchResult]:
        """
        Retrieve all stored memories for this user.

        Returns:
            All memory records (up to ChromaDB limit).
        """
        # Use a generic query to get all memories
        return self.retrieve_relevant(
            query="health medical condition symptom",
            top_k=100,
        )

    def delete_all_memories(self) -> None:
        """Delete all memories for this user."""
        memories = self.get_all_memories()
        if memories:
            collection = self.vector_store.get_or_create_collection(
                self._collection
            )
            ids = [m.doc_id for m in memories]
            collection.delete(ids=ids)
            log.warning(
                "Deleted {} memories for user={}",
                len(ids),
                self.user_id,
            )
