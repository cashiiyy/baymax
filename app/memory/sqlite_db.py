"""
BAYMAX AI – SQLite User Database
==================================
Persistent, structured storage for:
  - User profiles
  - Conversation history
  - Session metadata

Uses SQLAlchemy Core (async-compatible via aiosqlite).

Usage:
    from app.memory.sqlite_db import UserDatabase
    db = UserDatabase()
    await db.initialize()
    await db.save_message(user_id="user1", role="user", content="Hello")
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


class UserDatabase:
    """
    SQLite-based persistent storage for BAYMAX user data.

    Tables:
        users:         User profiles and metadata.
        conversations: Individual conversation turn records.
        sessions:      Session start/end tracking.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        from config import settings

        self.db_path = db_path or str(settings.SQLITE_DB_PATH)
        self._engine = None
        self._initialized = False
        log.info("UserDatabase configured | path={}", self.db_path)

    # ── Initialization ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create all tables if they don't exist."""
        if self._initialized:
            return

        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id     TEXT PRIMARY KEY,
                    name        TEXT NOT NULL DEFAULT 'User',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    metadata    TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content     TEXT NOT NULL,
                    emotion     TEXT,
                    timestamp   TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT,
                    turn_count  INTEGER DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE INDEX IF NOT EXISTS idx_conv_user
                    ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_session
                    ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conv_timestamp
                    ON conversations(timestamp);
            """)
            await db.commit()

        self._initialized = True
        log.info("UserDatabase tables initialized")

    # ── User Management ───────────────────────────────────────────────────────

    async def get_or_create_user(
        self,
        user_id: str,
        name: str = "User",
    ) -> dict:
        """
        Get an existing user or create a new one.

        Args:
            user_id: Unique user identifier.
            name:    Display name for new users.

        Returns:
            User record dict.
        """
        import aiosqlite
        import json

        now = _utcnow()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT OR IGNORE INTO users (user_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, name, now, now),
            )
            await db.commit()
            cursor = await db.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            return dict(row)

    async def update_user_metadata(
        self, user_id: str, metadata: dict
    ) -> None:
        """Update arbitrary metadata for a user."""
        import aiosqlite
        import json

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET metadata = ?, updated_at = ? WHERE user_id = ?",
                (json.dumps(metadata), _utcnow(), user_id),
            )
            await db.commit()

    # ── Conversation History ──────────────────────────────────────────────────

    async def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        emotion: Optional[str] = None,
    ) -> int:
        """
        Persist a conversation message.

        Args:
            user_id:    User identifier.
            session_id: Session identifier.
            role:       'user', 'assistant', or 'system'.
            content:    Message text.
            emotion:    Optional detected emotion label.

        Returns:
            Inserted row ID.
        """
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO conversations
                    (user_id, session_id, role, content, emotion, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, session_id, role, content, emotion, _utcnow()),
            )
            await db.execute(
                """
                UPDATE sessions SET turn_count = turn_count + 1
                WHERE session_id = ?
                """,
                (session_id,),
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """
        Retrieve conversation history for a user.

        Args:
            user_id:    User identifier.
            session_id: If provided, filter to this session.
            limit:      Maximum number of messages to return.

        Returns:
            List of message dicts, oldest first.
        """
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if session_id:
                cursor = await db.execute(
                    """
                    SELECT * FROM conversations
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (user_id, session_id, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (user_id, limit),
                )

            rows = await cursor.fetchall()
            return [dict(r) for r in reversed(rows)]  # Chronological order

    async def get_recent_turns(
        self,
        user_id: str,
        session_id: str,
        n: int = 10,
    ) -> List[dict]:
        """Return the N most recent conversation turns for a session."""
        history = await self.get_conversation_history(
            user_id=user_id, session_id=session_id, limit=n
        )
        return history[-n:]

    # ── Session Management ────────────────────────────────────────────────────

    async def create_session(self, user_id: str, session_id: str) -> None:
        """Record the start of a new conversation session."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, user_id, started_at)
                VALUES (?, ?, ?)
                """,
                (session_id, user_id, _utcnow()),
            )
            await db.commit()

    async def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (_utcnow(), session_id),
            )
            await db.commit()

    async def delete_user_data(self, user_id: str) -> None:
        """Delete all data for a user (GDPR compliance)."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            await db.commit()
        log.warning("All data deleted for user_id={}", user_id)


def _utcnow() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.datetime.utcnow().isoformat() + "Z"
