"""
BAYMAX AI – Memory Routes
===========================
"""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.memory_manager import MemoryManager

router = APIRouter(prefix="/memory", tags=["Memory Engine"])


class MemoryRecord(BaseModel):
    memory_id: str
    text: str
    score: float


@router.get("/{user_id}", response_model=List[MemoryRecord])
async def get_memories(user_id: str):
    """Get all episodic memories for a user."""
    # We instantiate a temporary manager just to query Chroma
    mm = MemoryManager(user_id=user_id)
    try:
        results = mm.vector_memory.get_all_memories()
        return [
            MemoryRecord(
                memory_id=r.doc_id,
                text=r.text,
                score=r.score,
            )
            for r in results
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{user_id}")
async def clear_memories(user_id: str):
    """Clear all memory (short-term, DB, vector) for a user."""
    mm = MemoryManager(user_id=user_id)
    try:
        await mm.clear_user_data()
        return {"status": "success", "detail": f"All data cleared for {user_id}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
