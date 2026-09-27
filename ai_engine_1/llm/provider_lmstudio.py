"""
AI Engine 1 – LM Studio External LLM Provider
================================================
Client for OpenAI-compatible local AI inference server running LM Studio
(e.g., Qwen 29B parameter model).

Decouples LM Studio from core backend logic with configurable timeouts,
health check validation, streaming/non-streaming support, and error recovery.

Usage:
    provider = LMStudioProvider(base_url="http://127.0.0.1:1234", model_id="local-model")
    if await provider.is_available_async():
        response = await provider.generate("Hello, I am Baymax.")
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List, Generator

import httpx

from ai_engine_1.llm.provider_base import LLMProvider
from ai_engine_1.llm.llm_engine import LLMResponse
from ai_engine_1.config import settings

try:
    from backend.logging.logger import get_logger
    logger = get_logger("ai-engine-1-lmstudio")
except Exception:
    import logging
    logger = logging.getLogger("ai-engine-1-lmstudio")


class LMStudioProvider(LLMProvider):
    """
    OpenAI-compatible inference provider for LM Studio local server.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or getattr(settings, "lmstudio_base_url", "http://127.0.0.1:1234")).rstrip("/")
        self.model_id = model_id or getattr(settings, "lmstudio_model", "local-model")
        self.timeout = timeout or float(getattr(settings, "lmstudio_timeout", 60.0))

        logger.info(
            f"[LLM] LMStudioProvider initialized | base_url={self.base_url} "
            f"model={self.model_id} timeout={self.timeout}s"
        )

    # ── LLMProvider Interface ─────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Generate response via LM Studio /v1/chat/completions endpoint."""
        start_time = time.time()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        endpoint = f"{self.base_url}/v1/chat/completions"

        # Explicit connect & read timeouts to prevent hanging
        timeout_config = httpx.Timeout(
            connect=5.0,
            read=self.timeout,
            write=15.0,
            pool=5.0,
        )

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

                content = ""
                if "choices" in data and len(data["choices"]) > 0:
                    msg = data["choices"][0].get("message", {})
                    content = msg.get("content", "").strip()

                if not content:
                    raise RuntimeError(f"LM Studio returned empty content in response: {data}")

                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", usage.get("completion_tokens", 0))
                latency = (time.time() - start_time) * 1000

                return LLMResponse(
                    text=content,
                    model_used=f"LMStudio/{self.model_id}",
                    tokens_used=tokens_used,
                    latency_ms=round(latency, 2),
                    fallback_triggered=False,
                )

            except httpx.ConnectError as conn_err:
                logger.warning(f"[LLM] LM Studio connection refused at {self.base_url}: {conn_err}")
                raise ConnectionError(f"LM Studio server is unreachable at {self.base_url}") from conn_err

            except httpx.TimeoutException as time_err:
                logger.warning(f"[LLM] LM Studio request timed out after {self.timeout}s: {time_err}")
                raise TimeoutError(f"LM Studio request timed out after {self.timeout}s") from time_err

            except Exception as exc:
                logger.warning(f"[LLM] LM Studio inference error: {exc}")
                raise

    def is_available(self) -> bool:
        """Synchronous quick check (cached or fast socket/probe)."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/v1/models")
                return res.status_code == 200
        except Exception:
            return False

    async def is_available_async(self) -> bool:
        """Non-blocking async health check."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/v1/models")
                return res.status_code == 200
        except Exception:
            return False

    async def check_health(self) -> Dict[str, Any]:
        """Detailed health probe for LM Studio."""
        start_t = time.time()
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(f"{self.base_url}/v1/models")
                latency_ms = round((time.time() - start_t) * 1000, 2)
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    return {
                        "status": "online",
                        "base_url": self.base_url,
                        "configured_model": self.model_id,
                        "available_models": models,
                        "latency_ms": latency_ms,
                        "error": None,
                    }
                return {
                    "status": "degraded",
                    "base_url": self.base_url,
                    "http_status": res.status_code,
                    "error": res.text[:200],
                }
        except Exception as exc:
            return {
                "status": "offline",
                "base_url": self.base_url,
                "error": str(exc),
            }

    def provider_name(self) -> str:
        return f"LMStudio/{self.model_id}"

    def unload(self) -> None:
        """No local memory to unload for remote/external inference service."""
        pass
