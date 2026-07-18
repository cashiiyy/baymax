"""
BAYMAX AI – Medical Knowledge Routes
======================================
"""

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/medical", tags=["Medical Knowledge Base"])
rag = RAGPipeline()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


@router.post("/search", response_model=SearchResponse)
async def search_medical_knowledge(req: SearchRequest):
    """Semantic search across all medical datasets."""
    try:
        result = rag.search(query=req.query, top_k=req.top_k)
        items = [
            SearchResultItem(
                text=r.text,
                source=r.collection,
                score=r.score,
            )
            for r in result.results
        ]
        return SearchResponse(query=req.query, results=items)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rebuild_index")
async def rebuild_index():
    """Trigger a rebuild of the RAG ChromaDB from processed datasets."""
    try:
        status = rag.build(force_rebuild=True)
        return {"status": "success", "details": status.collection_counts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
