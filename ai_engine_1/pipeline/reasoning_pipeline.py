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
from app.rag.pipeline import RAGPipeline
from ai_engine_1.pipeline.observation_schema import (
    VisionObservation,
    ObservationReasoningResponse,
    assess_observation_risk,
    build_observation_prompt,
)

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
        tools: MedicalToolRegistry = None,
        rag_pipeline: RAGPipeline = None
    ):
        self.planner = planner or IntelligentQueryPlanner()
        self.retriever = retriever or AdvancedRAGRetriever()
        self.embedder = embedder or AdvancedEmbedder()
        self.llm = llm or ProductionLLMEngine()
        self.prompts = prompts or PromptManager()
        self.safety = safety or MedicalSafetyEngine()
        self.explainer = explainer or ExplainabilityEngine()
        self.tools = tools or MedicalToolRegistry()
        self.rag_pipeline = rag_pipeline or RAGPipeline()

    async def execute_async(self, query: str, user_history: str = "") -> ReasoningPipelineResponse:
        start_time = time.time()
        from backend.logging.logger import get_logger
        log = get_logger("ai-engine-1-pipeline")

        log.info(f"[Stage 1/9] Request received | query_len={len(query)}")

        # Stage 1 & 2: Intent Classification & Query Planning
        plan = self.planner.plan(query)
        log.info(f"[Stage 2/9] Query planned | intent={plan.intent} rag_required={plan.rag_required} emergency={plan.emergency_protocol}")

        # Stage 3: Tool Execution (if required)
        tool_results = []
        if plan.tool_required:
            for tool_name in plan.recommended_tools:
                if tool_name == "unit_convert" and "celsius" in query.lower():
                    res = self.tools.execute_tool("unit_converter", value=102, from_unit="Fahrenheit", to_unit="Celsius")
                    tool_results.append(res.dict())
            log.info(f"[Stage 3/9] Tool execution completed | tools_run={len(tool_results)}")
        else:
            log.info("[Stage 3/9] No external tools required")

        # Stage 4 & 5: RAG Retrieval & Evidence Ranking
        evidence_docs = []
        evidence_text = "No evidence documents retrieved."
        if plan.rag_required:
            log.info("[Stage 4/9] RAG retrieval started...")
            try:
                rag_res = self.rag_pipeline.search(query, top_k=3)
                evidence_text = rag_res.to_context_string() if rag_res else "No evidence documents retrieved."
                if rag_res and hasattr(rag_res, "results"):
                    for r in rag_res.results:
                        evidence_docs.append({
                            "content": r.text,
                            "source_file": r.metadata.get("source", r.collection),
                            "score": r.score,
                            "citation": f"[{r.metadata.get('source', r.collection)}]"
                        })
                log.info(f"[Stage 5/9] RAG retrieval completed | docs_found={len(evidence_docs)}")
            except Exception as rag_err:
                log.warning(f"[Stage 5/9] RAG retrieval warning ({rag_err}), continuing without RAG context.")
                evidence_text = "No evidence documents retrieved."
        else:
            log.info("[Stage 4-5/9] RAG not required for this query intent")

        # Stage 6: LLM Reasoning
        assembled_prompt = self.prompts.assemble_reasoning_prompt(
            query=query,
            evidence_context=evidence_text,
            patient_memory=user_history,
            intent=plan.intent
        )
        system_prompt = self.prompts.get_prompt("system")
        log.info(f"[Stage 6/9] Prompt constructed | prompt_chars={len(assembled_prompt)} | LLM request starting via {self.llm.active_provider_name}...")

        llm_res = await self.llm.generate_async(assembled_prompt, system_prompt=system_prompt)
        log.info(f"[Stage 6/9] LLM response received | model={llm_res.model_used} latency={llm_res.latency_ms}ms tokens={llm_res.tokens_used}")

        # Stage 7: Safety Validation
        log.info("[Stage 7/9] Validating clinical safety...")
        safety_report = self.safety.validate_safety(query, llm_res.text)

        # Stage 8: Confidence Estimation & Explainability
        log.info("[Stage 8/9] Calculating confidence & citations...")
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
        log.info(f"[Stage 9/9] Final response ready | total_latency={round(latency, 2)}ms")

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

    async def execute_observation_async(
        self,
        observation: VisionObservation,
    ) -> ReasoningPipelineResponse:
        """Process a structured observation from AI Engine 2.

        Flow:
            1. Deterministic safety/risk assessment (no LLM)
            2. Build observation prompt with fact/inference separation
            3. RAG retrieval for relevant medical context
            4. LLM reasoning for empathetic, contextualized response
            5. Safety validation of LLM output
            6. Structured response

        The LLM explains and contextualizes — it does NOT make
        safety-critical decisions (those are deterministic in step 1).
        """
        start_time = time.time()

        # Step 1: Deterministic risk assessment (pre-LLM)
        risk = assess_observation_risk(observation)

        # Step 2: Build the observation prompt
        obs_prompt = build_observation_prompt(observation, risk)

        # Step 3: RAG retrieval for relevant medical context
        search_query = f"{observation.event_type} {observation.facial_state} {observation.movement_state}"
        evidence_docs = []
        evidence_text = ""
        try:
            rag_res = self.rag_pipeline.search(search_query, top_k=3)
            evidence_text = rag_res.to_context_string()
            for r in rag_res.results:
                evidence_docs.append({
                    "content": r.text,
                    "source_file": r.metadata.get("source", r.collection),
                    "score": r.score,
                    "citation": f"[{r.metadata.get('source', r.collection)}]"
                })
        except Exception:
            pass  # RAG is optional for observations

        if evidence_docs:
            obs_prompt += f"\n\n## RELEVANT MEDICAL CONTEXT (from knowledge base)\n{evidence_text}"

        # Step 4: LLM reasoning
        system_prompt = self.prompts.get_prompt("system")
        llm_res = await self.llm.generate_async(obs_prompt, system_prompt=system_prompt)

        # Step 5: Safety validation
        safety_report = self.safety.validate_safety(search_query, llm_res.text)

        # Override safety flags from deterministic assessment
        if risk.requires_immediate_action:
            safety_report.emergency_detected = True
            if "Immediate action required by deterministic safety rules" not in safety_report.warnings:
                safety_report.warnings.append("Immediate action required by deterministic safety rules")

        # Step 6: Build structured response
        confidence_info = self.explainer.calculate_confidence(
            retrieved_docs=evidence_docs,
            safety_warnings=safety_report.warnings,
            query=search_query,
        )

        plan = ExecutionPlan(
            query=search_query,
            intent="observation",
            rag_required=True,
            emergency_protocol=risk.requires_immediate_action,
            confidence_expected=observation.confidence,
        )

        final_answer = llm_res.text
        if risk.requires_immediate_action and "emergency" not in final_answer.lower():
            final_answer = (
                "🚨 **IMMEDIATE ATTENTION REQUIRED** 🚨\n\n"
                + final_answer
            )

        latency = (time.time() - start_time) * 1000

        return ReasoningPipelineResponse(
            query=search_query,
            response=final_answer,
            plan=plan,
            confidence=confidence_info,
            safety=safety_report,
            model_used=llm_res.model_used,
            total_latency_ms=round(latency, 2),
        )

reasoning_pipeline = MedicalReasoningPipeline()
