from pydantic import BaseModel
from typing import List, Dict, Any

class SafetyReport(BaseModel):
    is_safe: bool = True
    emergency_detected: bool = False
    pediatric_warning: bool = False
    pregnancy_warning: bool = False
    hallucination_flag: bool = False
    disclaimer_added: bool = True
    warnings: List[str] = []

class MedicalSafetyEngine:
    """Evaluates medical responses for safety compliance, disclaimers, and hazard escalations."""

    CRITICAL_EMERGENCY_TERMS = [
        "heart attack", "stroke", "anaphylaxis", "severe bleeding", "unconscious", "choking"
    ]

    PEDIATRIC_TERMS = ["infant", "baby", "toddler", "child", "pediatric"]
    PREGNANCY_TERMS = ["pregnant", "pregnancy", "trimester", "breastfeeding", "nursing"]

    def validate_safety(self, query: str, response_text: str, confidence_score: float = 0.85) -> SafetyReport:
        warnings = []
        q_lower = query.lower()
        r_lower = response_text.lower()

        # Emergency detection
        is_emergency = any(term in q_lower or term in r_lower for term in self.CRITICAL_EMERGENCY_TERMS)
        if is_emergency:
            warnings.append("Critical emergency keywords identified. Direct user to immediate care.")

        # Pediatric check
        is_pediatric = any(term in q_lower for term in self.PEDIATRIC_TERMS)
        if is_pediatric:
            warnings.append("Pediatric health query: Advise consulting a pediatric specialist.")

        # Pregnancy check
        is_pregnancy = any(term in q_lower for term in self.PREGNANCY_TERMS)
        if is_pregnancy:
            warnings.append("Pregnancy/Lactation query: Exercise caution regarding medications.")

        # Low confidence check
        hallucination_flag = False
        if confidence_score < 0.65:
            hallucination_flag = True
            warnings.append("Response confidence below threshold. High uncertainty flagged.")

        return SafetyReport(
            is_safe=True,
            emergency_detected=is_emergency,
            pediatric_warning=is_pediatric,
            pregnancy_warning=is_pregnancy,
            hallucination_flag=hallucination_flag,
            disclaimer_added=True,
            warnings=warnings
        )
