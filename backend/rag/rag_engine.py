from typing import List, Dict, Any

MEDICAL_DISCLAIMER = (
    "\n\n*Disclaimer: B.A.Y.M.A.X. provides educational guidance and information assistance only, "
    "not medical diagnoses or prescriptions. If you are experiencing a medical emergency or severe symptoms, "
    "please consult a qualified healthcare professional immediately.*"
)

class RAGEngine:
    """Orchestrates query intent, retrieval context, and medical safety disclaimers."""

    def __init__(self, retriever=None):
        self.retriever = retriever

    def format_prompt(self, query: str, context_docs: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"- {d.get('content', '')}" for d in context_docs]) if context_docs else "No specific documents found."
        
        prompt = (
            "You are BAYMAX, a compassionate local medical information assistant.\n"
            "Answer the user's question accurately using the provided medical context.\n"
            "Always include emergency warning signs if symptoms sound urgent.\n\n"
            f"Medical Context:\n{context_str}\n\n"
            f"User Query: {query}\n"
        )
        return prompt

    def enforce_safety(self, response_text: str) -> str:
        """Ensures educational disclaimer is present on medical advice outputs."""
        if MEDICAL_DISCLAIMER.strip() not in response_text:
            response_text += MEDICAL_DISCLAIMER
        return response_text
