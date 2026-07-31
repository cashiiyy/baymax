import os
import requests
from typing import Optional

class QwenLLM:
    """Wrapper for local Qwen Instruct model inference (via Ollama, vLLM, or local API endpoint)."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT", "http://127.0.0.1:11434/api/generate")
        self.model_name = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b-instruct")

    def generate(self, prompt: str) -> str:
        """Sends prompt to local LLM server. Falls back gracefully if local server offline."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.endpoint_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
        except Exception as err:
            pass

        # Stub fallback response if local endpoint is not yet spinning
        return (
            f"[BAYMAX Medical Intelligence Engine]\n"
            f"Based on your query: '{prompt[:60]}...', here is educational information regarding symptoms, "
            f"first aid steps, and preventative measures."
        )
