"""Tests for the Memory Engine."""
import pytest
import asyncio
from app.memory.sqlite_db import UserDatabase
from app.memory.short_term import ShortTermMemory

@pytest.mark.asyncio
async def test_sqlite_db(tmp_path):
    db_path = tmp_path / "test.db"
    db = UserDatabase(db_path=str(db_path))
    await db.initialize()
    
    user = await db.get_or_create_user("user123", "Test User")
    assert user["name"] == "Test User"
    
    await db.create_session("user123", "sess1")
    
    msg_id = await db.save_message("user123", "sess1", "user", "Hello", "happy")
    assert msg_id > 0
    
    history = await db.get_recent_turns("user123", "sess1", n=5)
    assert len(history) == 1
    assert history[0]["content"] == "Hello"
    assert history[0]["emotion"] == "happy"

def test_short_term_memory():
    mem = ShortTermMemory("u1", "s1", window=2)
    mem.add_turn("user", "msg1")
    mem.add_turn("assistant", "msg2")
    mem.add_turn("user", "msg3")
    
    assert mem.turn_count == 2
    turns = mem.get_turns()
    assert turns[0].content == "msg2"
    assert turns[1].content == "msg3"
