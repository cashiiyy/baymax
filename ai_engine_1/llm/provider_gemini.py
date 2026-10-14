"""
AI Engine 1 - Gemini Primary Provider
"""
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

from ai_engine_1.llm.provider_base import LLMProvider
from ai_engine_1.llm.llm_engine import LLMResponse
from ai_engine_1.config import settings

try:
    from backend.logging.logger import get_logger
    logger = get_logger("ai-engine-1-gemini")
except Exception:
    import logging
    logger = logging.getLogger("ai-engine-1-gemini")


class GeminiProvider(LLMProvider):
    """
    Primary LLM provider backed by the Gemini API.

    Raises:
        ConnectionError  - API key missing / network unreachable.
        TimeoutError     - Request exceeded timeout.
        RuntimeError     - Empty/malformed response from Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = (
            api_key
            or getattr(settings, "gemini_api_key", None)
            or os.getenv("GEMINI_API_KEY", "")
        )
        self.model_id = (
            model_id
            or getattr(settings, "gemini_model", "gemini-1.5-flash-latest")
        )
        self.timeout = timeout or float(
            getattr(settings, "gemini_timeout", 30.0)
        )

        if not self.api_key:
            logger.warning(
                "[LLM][Gemini] GEMINI_API_KEY not set - provider will fail at runtime."
            )
        else:
            logger.info(
                f"[LLM] GeminiProvider initialized | model={self.model_id} timeout={self.timeout}s"
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        if not self.api_key:
            raise ConnectionError(
                "GEMINI_API_KEY is not configured. Add it to .env and restart."
            )

        import asyncio
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model = genai.GenerativeModel(
            self.model_id,
            generation_config=generation_config,
            system_instruction=system_prompt or None,
        )

        start_time = time.time()
        logger.info(
            f"[LLM][Gemini] Sending request | model={self.model_id} max_tokens={max_tokens}"
        )

        try:
            response = await asyncio.wait_for(
                model.generate_content_async(prompt),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as te:
            raise TimeoutError(
                f"Gemini inference timed out after {self.timeout}s."
            ) from te
        except Exception as exc:
            err_str = str(exc)
            if any(k in err_str.lower() for k in ("connect", "network", "unreachable", "refused")):
                raise ConnectionError(f"Gemini API unreachable: {exc}") from exc
            raise

        text = response.text.strip() if response.text else ""
        if not text:
            raise RuntimeError(
                f"Gemini returned an empty response for model={self.model_id}."
            )

        latency = (time.time() - start_time) * 1000
        tokens_used = 0
        try:
            usage = response.usage_metadata
            tokens_used = (usage.total_token_count or 0)
        except Exception:
            pass

        logger.info(
            f"[LLM][Gemini] Response received | tokens={tokens_used} latency={round(latency, 1)}ms"
        )
        return LLMResponse(
            text=text,
            model_used=f"Gemini/{self.model_id}",
            tokens_used=tokens_used,
            latency_ms=round(latency, 2),
            fallback_triggered=False,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def is_available_async(self) -> bool:
        return bool(self.api_key)

    async def check_health(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "offline", "provider": "Gemini", "error": "GEMINI_API_KEY not set"}
        return {
            "status": "configured",
            "provider": "Gemini",
            "model": self.model_id,
            "error": None,
        }

    def provider_name(self) -> str:
        return f"Gemini/{self.model_id}"

    def unload(self) -> None:
        pass
