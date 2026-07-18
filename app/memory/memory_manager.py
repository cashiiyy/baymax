"""
BAYMAX AI – Memory Manager
============================
Unified interface combining all memory layers:
  - ShortTermMemory (in-memory circular buffer)
  - UserDatabase (SQLite persistent history)
  - VectorMemory (ChromaDB episodic memory)

Provides a single clean API for all memory operations.

Usage:
    from app.memory.memory_manager import MemoryManager
    mm = MemoryManager(user_id="user1", session_id="sess1")
    await mm.initialize()
    await mm.record_turn("user", "I feel dizzy")
    context = await mm.get_full_context("dizzy headache")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from app.memory.short_term import ShortTermMemory, ConversationTurn
from app.memory.sqlite_db import UserDatabase
from app.memory.vector_memory import VectorMemory
from app.rag.embedder import RAGEmbedder
from app.rag.vector_store import MedicalVectorStore
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FullMemoryContext:
    """
    Aggregated memory context for LLM prompt construction.

    Attributes:
        recent_turns:      Recent conversation turns from short-term memory.
        relevant_memories: Semantically retrieved episodic memories.
        user_profile:      User profile from SQLite.
        short_term_string: Pre-formatted short-term memory string.
        memory_string:     Pre-formatted episodic memory string.
    """
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    relevant_memories: list = field(default_factory=list)
    user_profile: dict = field(default_factory=dict)
    short_term_string: str = ""
    memory_string: str = ""


class MemoryManager:
    """
    Unified memory layer for BAYMAX.

    Orchestrates all three memory types to provide seamless context
    retrieval before each LLM call.

    Attributes:
        user_id:       User identifier.
        session_id:    Current session identifier (auto-generated if None).
        short_term:    ShortTermMemory instance.
        db:            UserDatabase (SQLite) instance.
        vector_memory: VectorMemory (ChromaDB) instance.
    """

    def __init__(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        db: Optional[UserDatabase] = None,
        embedder: Optional[RAGEmbedder] = None,
        vector_store: Optional[MedicalVectorStore] = None,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())

        # Shared embedder/store instances (injected to avoid duplicating models)
        _embedder = embedder or RAGEmbedder()
        _store = vector_store or MedicalVectorStore()

        self.short_term = ShortTermMemory(
            user_id=user_id,
            session_id=self.session_id,
        )
        self.db = db or UserDatabase()
        self.vector_memory = VectorMemory(
            user_id=user_id,
            embedder=_embedder,
            vector_store=_store,
        )
        self._initialized = False
        log.info(
            "MemoryManager created | user={} session={}",
            user_id,
            self.session_id[:8],
        )

    # ── Initialization ────────────────────────────────────────────────────────

    async def initialize(self, user_name: str = "User") -> None:
        """
        Initialize the memory manager.

        Creates user record and session entry in SQLite.

        Args:
            user_name: Display name for new users.
        """
        if self._initialized:
            return
        await self.db.initialize()
        await self.db.get_or_create_user(self.user_id, name=user_name)
        await self.db.create_session(self.user_id, self.session_id)
        self._initialized = True
        log.info("MemoryManager initialized for user={}", self.user_id)

    # ── Recording Turns ───────────────────────────────────────────────────────

    async def record_turn(
        self,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        store_as_memory: bool = False,
    ) -> None:
        """
        Record a conversation turn in all memory layers.

        Args:
            role:             'user' or 'assistant'.
            content:          Message content.
            emotion:          Detected emotion label.
            store_as_memory:  If True, also store in episodic vector memory.
        """
        # Short-term memory (always)
        self.short_term.add_turn(role, content, emotion)

        # SQLite persistence (always)
        await self.db.save_message(
            user_id=self.user_id,
            session_id=self.session_id,
            role=role,
            content=content,
            emotion=emotion,
        )

        # Episodic vector memory (optional, for important user facts)
        if store_as_memory and role == "user":
            memory_text = f"User said: {content}"
            if emotion:
                memory_text += f" (felt {emotion})"
            self.vector_memory.store_memory(
                content=memory_text,
                session_id=self.session_id,
                importance=0.6,
            )

        log.debug(
            "Turn recorded | role={} session={}",
            role,
            self.session_id[:8],
        )

    async def record_user_fact(self, fact: str) -> str:
        """
        Store an important user fact directly in episodic memory.

        Examples:
            - "User has a peanut allergy"
            - "User is 45 years old and diabetic"

        Args:
            fact: Fact string to store.

        Returns:
            Memory ID.
        """
        return self.vector_memory.store_memory(
            content=fact,
            session_id=self.session_id,
            importance=0.9,
        )

    # ── Context Retrieval ─────────────────────────────────────────────────────

    async def get_full_context(
        self,
        query: str,
        recent_n: int = 10,
    ) -> FullMemoryContext:
        """
        Retrieve all relevant memory context for a given query.

        Combines:
            - Recent short-term turns
            - Semantically retrieved episodic memories
            - User profile

        Args:
            query:    Current user query for semantic retrieval.
            recent_n: Number of recent turns to include.

        Returns:
            FullMemoryContext ready for prompt injection.
        """
        # Short-term context
        recent_turns = self.short_term.get_turns(n=recent_n)
        short_term_string = self.short_term.get_context_string(n=recent_n)

        # Episodic memory retrieval
        relevant_memories = self.vector_memory.retrieve_relevant(query)
        memory_string = self.vector_memory.retrieve_context_string(query)

        # User profile from SQLite
        user_profile = await self.db.get_or_create_user(self.user_id)

        return FullMemoryContext(
            recent_turns=recent_turns,
            relevant_memories=relevant_memories,
            user_profile=user_profile,
            short_term_string=short_term_string,
            memory_string=memory_string,
        )

    def get_llm_messages(self, n: int = 10) -> List[dict]:
        """
        Get recent conversation as LLM-formatted message list.

        Args:
            n: Number of recent turns to include.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        return self.short_term.get_llm_messages(n)

    async def end_session(
        self,
        summary: Optional[str] = None,
    ) -> None:
        """
        End the current session.

        Optionally stores a session summary in episodic memory.

        Args:
            summary: Auto-generated or manual session summary.
        """
        await self.db.end_session(self.session_id)

        if summary:
            self.vector_memory.store_conversation_summary(
                summary=summary,
                session_id=self.session_id,
            )

        log.info(
            "Session ended | session={} turns={}",
            self.session_id[:8],
            self.short_term.turn_count,
        )

    async def clear_user_data(self) -> None:
        """Delete all memory data for this user."""
        self.short_term.clear()
        await self.db.delete_user_data(self.user_id)
        self.vector_memory.delete_all_memories()
        log.warning("All memory cleared for user={}", self.user_id)
