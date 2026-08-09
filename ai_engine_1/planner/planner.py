from pydantic import BaseModel
from typing import List, Dict, Any

class ExecutionPlan(BaseModel):
    query: str
    intent: str # emergency / symptom_inquiry / first_aid / document_ocr / general
    rag_required: bool = True
    tool_required: bool = False
    recommended_tools: List[str] = []
    emergency_protocol: bool = False
    confidence_expected: float = 0.85

class IntelligentQueryPlanner:
    """Analyzes user query to create structured execution plan before LLM inference."""

    EMERGENCY_KEYWORDS = [
        "chest pain", "cannot breathe", "severe bleeding", "unconscious",
        "stroke", "heart attack", "poison", "suicide", "choking"
    ]

    CHAT_KEYWORDS = [
        "hi", "hello", "hey", "greetings", "whats up" ,"howdy", "sup", "good morning", "good afternoon", "good evening"
    ]

    TOOL_KEYWORDS = {
        "bmi": ["bmi", "body mass index", "weight height ratio"],
        "bsa": ["bsa", "body surface area"],
        "unit_convert": ["convert", "fahrenheit", "celsius", "mg/dl"],
        "drug_lookup": ["interaction", "side effect", "contraindication", "medication"]
    }

    def plan(self, query: str) -> ExecutionPlan:
        q_lower = query.lower().strip()

        # Check casual greetings / chitchat first
        is_chitchat = q_lower in self.CHAT_KEYWORDS or any(q_lower.startswith(prefix + " ") for prefix in self.CHAT_KEYWORDS)
        if is_chitchat:
            return ExecutionPlan(
                query=query,
                intent="general_chat",
                rag_required=False,
                tool_required=False,
                emergency_protocol=False,
                confidence_expected=0.5
            )

        # Check Emergency
        is_emergency = any(kw in q_lower for kw in self.EMERGENCY_KEYWORDS)
        if is_emergency:
            return ExecutionPlan(
                query=query,
                intent="emergency",
                rag_required=True,
                tool_required=False,
                emergency_protocol=True,
                confidence_expected=0.95
            )

        # Check Tools
        needed_tools = []
        for tool, kws in self.TOOL_KEYWORDS.items():
            if any(kw in q_lower for kw in kws):
                needed_tools.append(tool)

        intent = "symptom_inquiry"
        if "first aid" in q_lower or "cpr" in q_lower or "burn" in q_lower:
            intent = "first_aid"

        return ExecutionPlan(
            query=query,
            intent=intent,
            rag_required=True,
            tool_required=len(needed_tools) > 0,
            recommended_tools=needed_tools,
            emergency_protocol=False,
            confidence_expected=0.85
        )
