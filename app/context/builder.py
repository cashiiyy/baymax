"""
BAYMAX AI – Context Builder
============================
Combines all pipeline inputs into a structured prompt for the LLM:
  - User transcript
  - Detected emotion
  - Short-term conversation history
  - User profile (from SQLite)
  - Retrieved episodic memories
  - Retrieved medical RAG documents

Usage:
    from app.context.builder import ContextBuilder
    builder = ContextBuilder(rag_pipeline, memory_manager)
    prompt = await builder.build(
        user_message="I have a sore throat",
        emotion_result=emotion,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.context.prompt_templates import (
    build_context_prompt,
    build_chat_messages,
    format_system_prompt,
)
from app.emotion.deepface_engine import EmotionResult
from app.memory.memory_manager import FullMemoryContext, MemoryManager
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import MedicalRetrievalResult
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BuiltContext:
    """
    The fully assembled context for an LLM call.

    Attributes:
        system_prompt:       BAYMAX system instructions.
        chat_messages:       Full chat messages list for HF chat template.
        context_block:       The context injection string (for logging/debugging).
        medical_results:     Raw RAG retrieval results.
        memory_context:      Full memory context.
        user_message:        Original user message.
        emotion_label:       Detected emotion.
        emotion_confidence:  Emotion confidence score.
        is_distressed:       Whether user is distressed.
    """
    system_prompt: str
    chat_messages: List[dict]
    context_block: str
    medical_results: Optional[MedicalRetrievalResult] = None
    memory_context: Optional[FullMemoryContext] = None
    user_message: str = ""
    emotion_label: str = "neutral"
    emotion_confidence: float = 0.5
    is_distressed: bool = False

    @property
    def has_medical_context(self) -> bool:
        return (
            self.medical_results is not None
            and len(self.medical_results.results) > 0
        )


class ContextBuilder:
    """
    Assembles the complete LLM context from all pipeline inputs.

    This is the central hub that connects:
        RAG Pipeline → Medical knowledge
        MemoryManager → Conversation history + episodic memory
        EmotionResult → Detected emotional state
        User message → Current query

    Attributes:
        rag_pipeline:    RAGPipeline instance for medical retrieval.
        memory_manager:  MemoryManager for conversation + episodic memory.
        max_recent_turns: Number of recent turns to include in history.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_manager: Optional[MemoryManager] = None,
        max_recent_turns: int = 10,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.memory_manager = memory_manager
        self.max_recent_turns = max_recent_turns
        log.info("ContextBuilder initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    async def build(
        self,
        user_message: str,
        emotion_result: Optional[EmotionResult] = None,
        user_id: str = "default_user",
        user_name: str = "User",
        rag_top_k: Optional[int] = None,
    ) -> BuiltContext:
        """
        Build the complete LLM context for a user message.

        Steps:
            1. Extract emotion state
            2. Retrieve medical knowledge from RAG
            3. Retrieve memory context
            4. Assemble full prompt

        Args:
            user_message:   Current user text input.
            emotion_result: Emotion detection result from the camera.
            user_id:        User identifier for memory retrieval.
            user_name:      User display name.
            rag_top_k:      Override RAG result count.

        Returns:
            BuiltContext ready for LLM inference.
        """
        log.info(
            "Building context | user={} | message='{}'",
            user_id,
            user_message[:60],
        )

        # ── Step 1: Extract emotion state ─────────────────────────────────────
        emotion_label, emotion_confidence, is_distressed = self._extract_emotion(
            emotion_result
        )

        # ── Step 2: Medical RAG retrieval ─────────────────────────────────────
        medical_results: Optional[MedicalRetrievalResult] = None
        medical_context_str = ""

        if self.rag_pipeline is not None:
            try:
                medical_results = self.rag_pipeline.search(
                    query=user_message,
                    top_k=rag_top_k,
                )
                medical_context_str = medical_results.context
                log.debug(
                    "RAG retrieved {} docs",
                    len(medical_results.results),
                )
            except Exception as exc:
                log.warning("RAG retrieval failed: {}", exc)

        # ── Step 3: Memory context ────────────────────────────────────────────
        memory_ctx: Optional[FullMemoryContext] = None
        memory_context_str = ""
        conversation_history: List[dict] = []

        if self.memory_manager is not None:
            try:
                memory_ctx = await self.memory_manager.get_full_context(
                    query=user_message,
                    recent_n=self.max_recent_turns,
                )
                memory_context_str = (
                    f"{memory_ctx.short_term_string}\n\n"
                    f"{memory_ctx.memory_string}"
                )
                conversation_history = self.memory_manager.get_llm_messages(
                    n=self.max_recent_turns
                )

                # Update user_name from profile if available
                profile_name = memory_ctx.user_profile.get("name")
                if profile_name and profile_name != "User":
                    user_name = profile_name

            except Exception as exc:
                log.warning("Memory retrieval failed: {}", exc)

        # ── Step 4: Assemble context block ────────────────────────────────────
        context_block = build_context_prompt(
            user_message=user_message,
            user_name=user_name,
            user_id=user_id,
            emotion=emotion_label,
            emotion_confidence=emotion_confidence,
            is_distressed=is_distressed,
            medical_context=medical_context_str,
            memory_context=memory_context_str,
        )

        # ── Step 5: Build chat messages for LLM ───────────────────────────────
        system_prompt = format_system_prompt()
        chat_messages = build_chat_messages(
            system_prompt=context_block,  # Inject full context as system msg
            conversation_history=conversation_history,
            current_message=user_message,
        )

        result = BuiltContext(
            system_prompt=system_prompt,
            chat_messages=chat_messages,
            context_block=context_block,
            medical_results=medical_results,
            memory_context=memory_ctx,
            user_message=user_message,
            emotion_label=emotion_label,
            emotion_confidence=emotion_confidence,
            is_distressed=is_distressed,
        )

        log.info(
            "Context built | emotion={} | rag_docs={} | history_turns={}",
            emotion_label,
            len(medical_results.results) if medical_results else 0,
            len(conversation_history),
        )
        return result

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_emotion(
        emotion_result: Optional[EmotionResult],
    ) -> tuple[str, float, bool]:
        """
        Extract emotion state from an EmotionResult.

        Returns:
            Tuple of (label, confidence, is_distressed).
        """
        if emotion_result is None:
            return "neutral", 0.5, False

        smoothed = emotion_result.smoothed
        return (
            smoothed.dominant_emotion,
            smoothed.confidence,
            smoothed.is_distressed,
        )
