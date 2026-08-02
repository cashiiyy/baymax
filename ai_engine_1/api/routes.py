from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ai_engine_1.pipeline.reasoning_pipeline import reasoning_pipeline, ReasoningPipelineResponse
from ai_engine_1.embeddings.embedder import AdvancedEmbedder
from ai_engine_1.retriever.retriever import AdvancedRAGRetriever
from ai_engine_1.planner.planner import IntelligentQueryPlanner, ExecutionPlan
from ai_engine_1.observability.metrics import metrics_collector
from ai_engine_1.llm.llm_engine import ProductionLLMEngine

router = APIRouter(prefix="/engine1", tags=["AI Engine 1"])

embedder = AdvancedEmbedder()
retriever = AdvancedRAGRetriever()
planner = IntelligentQueryPlanner()
llm = ProductionLLMEngine()

class ReasonRequest(BaseModel):
    query: str
    user_history: Optional[str] = ""

class EmbedRequest(BaseModel):
    text: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class SummarizeRequest(BaseModel):
    text: str

class ClassifyRequest(BaseModel):
    query: str

@router.post("/reason", response_model=ReasoningPipelineResponse)
async def reason_endpoint(req: ReasonRequest):
    try:
        metrics_collector.record_request(success=True)
        return await reasoning_pipeline.execute_async(req.query, req.user_history)
    except Exception as e:
        metrics_collector.record_request(success=False)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
async def retrieve_endpoint(req: RetrieveRequest):
    q_vec = await embedder.encode_async(req.query)
    results = retriever.search_hybrid(q_vec, req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}

@router.post("/embed")
async def embed_endpoint(req: EmbedRequest):
    vec = await embedder.encode_async(req.text)
    return {"text": req.text, "vector": vec.tolist(), "dimension": len(vec)}

@router.post("/summarize")
async def summarize_endpoint(req: SummarizeRequest):
    prompt = f"Summarize the following text into concise medical takeaways:\n\n{req.text}"
    res = await llm.generate_async(prompt)
    return {"summary": res.text, "model_used": res.model_used}

@router.post("/classify")
async def classify_endpoint(req: ClassifyRequest):
    plan = planner.plan(req.query)
    return {"query": req.query, "intent": plan.intent, "emergency": plan.emergency_protocol}

@router.post("/plan", response_model=ExecutionPlan)
async def plan_endpoint(req: ClassifyRequest):
    return planner.plan(req.query)

@router.get("/health")
@router.post("/health")
async def health_endpoint():
    return {"status": "online", "engine": "AI Engine 1 Medical Intelligence"}

@router.get("/metrics")
@router.post("/metrics")
async def metrics_endpoint():
    return metrics_collector.get_summary()

@router.get("/status")
@router.post("/status")
async def status_endpoint():
    return {
        "engine": "AI Engine 1",
        "status": "ready",
        "primary_llm": llm.primary_model,
        "fallback_llm": llm.fallback_model,
        "embedding_model": embedder.model_name
    }
