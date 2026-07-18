"""
BAYMAX AI – Short-Term Memory
==============================
In-memory circular buffer holding recent conversation turns for a session.
Provides instant access to recent context without disk I/O.

Usage:
    from app.memory.short_term import ShortTermMemory
    mem = ShortTermMemory(user_id="user1", session_id="sess1")
    mem.add_turn("user", "I have a headache")
    mem.add_turn("assistant", "I understand. Let me check that.")
    context = mem.get_context_string()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str          # 'user' or 'assistant'
    content: str
    emotion: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "emotion": self.emotion,
            "timestamp": self.timestamp,
        }

    def to_llm_message(self) -> dict:
        """Format as a chat message dict for LLM input."""
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """
    Circular buffer of recent conversation turns (in-memory).

    Provides the recent conversation window for context building.
    Does NOT persist to disk – use SQLite for long-term persistence.

    Attributes:
        user_id:    User identifier.
        session_id: Session identifier.
        window:     Maximum number of turns to retain.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        window: Optional[int] = None,
    ) -> None:
        from config import settings

        self.user_id = user_id
        self.session_id = session_id
        self.window = window or settings.SHORT_TERM_WINDOW
        self._turns: Deque[ConversationTurn] = deque(maxlen=self.window)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(
        self,
        role: str,
        content: str,
        emotion: Optional[str] = None,
    ) -> None:
        """
        Add a new conversation turn to memory.

        Args:
            role:    'user' or 'assistant'.
            content: Turn text content.
            emotion: Optional detected emotion label.
        """
        turn = ConversationTurn(role=role, content=content, emotion=emotion)
        self._turns.append(turn)

    def get_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        Return the most recent N turns.

        Args:
            n: Number of turns to return. Returns all if None.

        Returns:
            List of ConversationTurn in chronological order.
        """
        turns = list(self._turns)
        if n is not None:
            return turns[-n:]
        return turns

    def get_llm_messages(self, n: Optional[int] = None) -> List[dict]:
        """
        Format recent turns as a list of chat messages for the LLM.

        Args:
            n: Number of turns to include.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        return [t.to_llm_message() for t in self.get_turns(n)]

    def get_context_string(self, n: Optional[int] = None) -> str:
        """
        Format recent turns as a human-readable string for prompt injection.

        Args:
            n: Number of recent turns to include.

        Returns:
            Formatted conversation string.
        """
        turns = self.get_turns(n)
        if not turns:
            return "No previous conversation."

        lines = ["=== RECENT CONVERSATION ==="]
        for turn in turns:
            role_label = "User" if turn.role == "user" else "BAYMAX"
            emotion_tag = f" [{turn.emotion}]" if turn.emotion else ""
            lines.append(f"{role_label}{emotion_tag}: {turn.content}")
        return "\n".join(lines)

    def get_last_user_message(self) -> Optional[str]:
        """Return the most recent user message content."""
        for turn in reversed(list(self._turns)):
            if turn.role == "user":
                return turn.content
        return None

    def clear(self) -> None:
        """Clear all turns from memory."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        """Return the number of turns currently in memory."""
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0
