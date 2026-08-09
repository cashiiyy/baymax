from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ai_engine_1.pipeline.reasoning_pipeline import reasoning_pipeline, ReasoningPipelineResponse
from ai_engine_1.embeddings.embedder import AdvancedEmbedder
from ai_engine_1.retriever.retriever import AdvancedRAGRetriever
from ai_engine_1.planner.planner import IntelligentQueryPlanner, ExecutionPlan
from ai_engine_1.observability.metrics import metrics_collector
from ai_engine_1.llm.llm_engine import ProductionLLMEngine
from ai_engine_1.pipeline.observation_schema import VisionObservation

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

# ── New Endpoints: AI Engine 2 Observation & LLM Status ──────────────────────

@router.post("/observation", response_model=ReasoningPipelineResponse)
async def observation_endpoint(obs: VisionObservation):
    """Process a structured observation from AI Engine 2.

    Accepts a VisionObservation event (e.g. possible_fall, distress_detected)
    and runs it through the full reasoning pipeline with deterministic safety
    rules applied before LLM reasoning.
    """
    try:
        metrics_collector.record_request(success=True)
        return await reasoning_pipeline.execute_observation_async(obs)
    except Exception as e:
        metrics_collector.record_request(success=False)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/llm-status")
async def llm_status_endpoint():
    """Return current LLM provider configuration and availability."""
    provider_name = llm.active_provider_name
    provider_type = llm._llm_provider_type

    status = {
        "provider_type": provider_type,
        "active_provider": provider_name,
        "omniroute_model": llm.primary_model,
        "omniroute_configured": bool(llm.omniroute_api_key),
    }

    # Check local Qwen provider availability if configured
    if provider_type == "qwen_local":
        try:
            local_provider = llm._get_qwen_local_provider()
            status["qwen_local"] = {
                "available": local_provider.is_available(),
                "model_id": local_provider.model_id,
                "model_loaded": local_provider._model is not None,
                "quantization": "4-bit NF4",
            }
        except Exception as e:
            status["qwen_local"] = {"available": False, "error": str(e)}

    return status

