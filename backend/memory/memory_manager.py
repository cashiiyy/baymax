from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.database.models import MemoryItem, Conversation

class MemoryManager:
    """Manages short-term conversation context and long-term saved facts."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_short_term_history(self, user_id: int, limit: int = 10) -> List[Dict[str, str]]:
        records = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.timestamp.desc())
            .limit(limit)
            .all()
        )
        records.reverse()
        return [{"role": r.role, "message": r.message} for r in records]

    def add_conversation_turn(self, user_id: int, role: str, message: str):
        turn = Conversation(user_id=user_id, role=role, message=message)
        self.db.add(turn)
        self.db.commit()

    def store_fact(self, user_id: int, key: str, value: str, category: str = "fact"):
        item = MemoryItem(user_id=user_id, key=key, value=value, category=category)
        self.db.add(item)
        self.db.commit()

    def get_user_facts(self, user_id: int) -> List[Dict[str, str]]:
        items = self.db.query(MemoryItem).filter(MemoryItem.user_id == user_id).all()
        return [{"category": i.category, "key": i.key, "value": i.value} for i in items]
