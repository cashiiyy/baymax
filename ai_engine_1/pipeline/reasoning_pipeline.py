import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ai_engine_1.planner.planner import IntelligentQueryPlanner, ExecutionPlan
from ai_engine_1.retriever.retriever import AdvancedRAGRetriever
from ai_engine_1.llm.llm_engine import ProductionLLMEngine
from ai_engine_1.prompts.prompt_manager import PromptManager
from ai_engine_1.safety.safety_engine import MedicalSafetyEngine, SafetyReport
from ai_engine_1.explainability.explainer import ExplainabilityEngine, ConfidenceScore
from ai_engine_1.tools.medical_tools import MedicalToolRegistry
from ai_engine_1.embeddings.embedder import AdvancedEmbedder

class ReasoningPipelineResponse(BaseModel):
    query: str
    response: str
    plan: ExecutionPlan
    confidence: ConfidenceScore
    safety: SafetyReport
    tool_results: List[Dict[str, Any]] = []
    model_used: str
    total_latency_ms: float

class MedicalReasoningPipeline:
    """9-Stage Structured Medical Reasoning Pipeline:
    Query -> Intent -> Entity Extraction -> RAG -> Evidence Ranking -> LLM Reasoning -> Safety -> Confidence -> Response
    """

    def __init__(
        self,
        planner: IntelligentQueryPlanner = None,
        retriever: AdvancedRAGRetriever = None,
        embedder: AdvancedEmbedder = None,
        llm: ProductionLLMEngine = None,
        prompts: PromptManager = None,
        safety: MedicalSafetyEngine = None,
        explainer: ExplainabilityEngine = None,
        tools: MedicalToolRegistry = None
    ):
        self.planner = planner or IntelligentQueryPlanner()
        self.retriever = retriever or AdvancedRAGRetriever()
        self.embedder = embedder or AdvancedEmbedder()
        self.llm = llm or ProductionLLMEngine()
        self.prompts = prompts or PromptManager()
        self.safety = safety or MedicalSafetyEngine()
        self.explainer = explainer or ExplainabilityEngine()
        self.tools = tools or MedicalToolRegistry()

    async def execute_async(self, query: str, user_history: str = "") -> ReasoningPipelineResponse:
        start_time = time.time()

        # Stage 1 & 2: Intent Classification & Query Planning
        plan = self.planner.plan(query)

        # Stage 3: Tool Execution (if required)
        tool_results = []
        if plan.tool_required:
            for tool_name in plan.recommended_tools:
                # Example unit conversion or abbreviation check
                if tool_name == "unit_convert" and "celsius" in query.lower():
                    res = self.tools.execute_tool("unit_converter", value=102, from_unit="Fahrenheit", to_unit="Celsius")
                    tool_results.append(res.dict())

        # Stage 4 & 5: RAG Retrieval & Evidence Ranking
        evidence_docs = []
        if plan.rag_required:
            q_vector = await self.embedder.encode_async(query)
            evidence_docs = self.retriever.search_hybrid(q_vector, query, top_k=3)

        evidence_text = self.retriever.format_evidence_block(evidence_docs)

        # Stage 6: LLM Reasoning
        assembled_prompt = self.prompts.assemble_reasoning_prompt(
            query=query,
            evidence_context=evidence_text,
            patient_memory=user_history
        )
        system_prompt = self.prompts.get_prompt("system")

        llm_res = await self.llm.generate_async(assembled_prompt, system_prompt=system_prompt)

        # Stage 7: Safety Validation
        safety_report = self.safety.validate_safety(query, llm_res.text)

        # Stage 8: Confidence Estimation & Explainability
        confidence_info = self.explainer.calculate_confidence(
            retrieved_docs=evidence_docs,
            safety_warnings=safety_report.warnings,
            query=query
        )

        # Stage 9: Final Response Generation
        final_answer = llm_res.text
        if "disclaimer" not in final_answer.lower():
            final_answer += (
                "\n\n*Disclaimer: Baymax provides educational guidance only. "
                "Consult a healthcare professional for medical emergencies.*"
            )

        latency = (time.time() - start_time) * 1000

        return ReasoningPipelineResponse(
            query=query,
            response=final_answer,
            plan=plan,
            confidence=confidence_info,
            safety=safety_report,
            tool_results=tool_results,
            model_used=llm_res.model_used,
            total_latency_ms=round(latency, 2)
        )

reasoning_pipeline = MedicalReasoningPipeline()
