import json
import os
from typing import Dict, Any, Optional

PROMPTS_DIR = os.path.dirname(__file__)

DEFAULT_PROMPTS = {
    "system": (
        "You are Baymax, a compassionate, friendly personal healthcare companion. "
        "Always call yourself 'Baymax' (never 'B.A.Y.M.A.X.'). "
        "Your responses MUST be small, concise, and informative—never overly descriptive, long, or verbose. "
        "Provide warm, empathetic, clear healthcare guidance in 2 to 3 brief sentences max."
    ),
    "medical": (
        "User Query: {query}\n"
        "Evidence Context: {evidence_context}\n"
        "Patient History: {patient_memory}\n\n"
        "Answer as Baymax. Keep the response small, concise, and informative (maximum 2-3 short sentences)."
    ),
    "safety": (
        "Validate the following proposed response for medical safety:\n"
        "Proposed Response:\n{response_text}\n\n"
        "Rules:\n"
        "- Disallow exact dosage recommendations for prescription drugs.\n"
        "- Verify emergency warning flags if severe symptoms are mentioned.\n"
        "- Ensure non-diagnostic educational disclaimer is present."
    ),
    "emergency": (
        "🚨 EMERGENCY WARNING SIGN DETECTED 🚨\n"
        "Query: {query}\n"
        "Immediate Action Required:\n"
        "1. Direct user to call emergency services (e.g., 911 / 108 / 112) immediately.\n"
        "2. Provide calm, immediate first-aid steps while waiting for paramedics.\n"
        "3. Do not delay emergency consultation."
    ),
    "followup": (
        "Based on previous discussion about {topic}, generate 2-3 helpful follow-up questions or self-monitoring tips for the user."
    ),
    "ocr": (
        "Extracted Medical Document Text:\n{ocr_text}\n\n"
        "Analyze this medical text, summarize key findings, explain medical terminology, and highlight any vital instructions."
    ),
    "vision": (
        "Medical Image Description / Analysis Findings:\n{image_findings}\n\n"
        "Correlate visual findings with potential health conditions, offering educational insights and recommending clinical evaluation."
    ),
    "memory": (
        "Summarize important long-term user health facts (allergies, conditions, preferences) from this conversation:\n{conversation_text}"
    )
}

class PromptManager:
    """Manages template assembly for System, Medical, Safety, Emergency, Followup, OCR, Vision, and Memory prompts."""

    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = templates_dir or PROMPTS_DIR
        self.prompts = dict(DEFAULT_PROMPTS)
        self._load_external_templates()

    def _load_external_templates(self):
        for key in DEFAULT_PROMPTS.keys():
            file_path = os.path.join(self.templates_dir, f"{key}_prompt.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "template" in data:
                            self.prompts[key] = data["template"]
                except Exception:
                    pass

    def get_prompt(self, template_name: str, **kwargs) -> str:
        template = self.prompts.get(template_name, DEFAULT_PROMPTS.get(template_name, ""))
        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def assemble_reasoning_prompt(
        self,
        query: str,
        evidence_context: str = "No specific RAG evidence found.",
        patient_memory: str = "None provided."
    ) -> str:
        system = self.get_prompt("system")
        medical = self.get_prompt("medical", query=query, evidence_context=evidence_context, patient_memory=patient_memory)
        return f"{system}\n\n{medical}"

prompt_manager = PromptManager()
