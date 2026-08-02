from typing import Optional
from ai_engine_1.llm.llm_engine import ProductionLLMEngine


class QwenLLM:
    """Wrapper preserving backward compatibility with legacy calls while leveraging ProductionLLMEngine."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.engine = ProductionLLMEngine()

    def generate(self, prompt: str) -> str:
        """Synchronous generate — returns plain text from the LLM engine."""
        response = self.engine.generate(prompt)
        return response.text
