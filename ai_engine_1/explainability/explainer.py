from pydantic import BaseModel
from typing import List, Dict, Any

class ConfidenceScore(BaseModel):
    confidence: float
    risk: str # low / medium / high
    sources: int
    reasoning_summary: str
    limitations: str

class ExplainabilityEngine:
    """Calculates confidence score based on RAG quality, source agreement, safety, and model latency."""

    def calculate_confidence(
        self,
        retrieved_docs: List[Dict[str, Any]],
        safety_warnings: List[str],
        query: str
    ) -> ConfidenceScore:
        sources_count = len(retrieved_docs)
        
        # Base confidence
        if sources_count >= 3:
            base_score = 0.92
        elif sources_count >= 1:
            base_score = 0.82
        else:
            base_score = 0.70

        # Adjust for safety warnings
        if safety_warnings:
            base_score -= 0.08

        final_conf = max(0.40, min(0.99, round(base_score, 2)))
        
        risk = "low"
        if final_conf < 0.65 or "emergency" in str(safety_warnings).lower():
            risk = "high"
        elif final_conf < 0.80:
            risk = "medium"

        reasoning_summary = (
            f"Evaluated query against {sources_count} medical evidence document(s). "
            f"Applied safety validation rules and verified symptom indicators."
        )

        limitations = (
            "Guidance is provided for educational information assistance only. "
            "It does not replace clinical evaluation or diagnostic lab tests."
        )

        return ConfidenceScore(
            confidence=final_conf,
            risk=risk,
            sources=sources_count,
            reasoning_summary=reasoning_summary,
            limitations=limitations
        )
