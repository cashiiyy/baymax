"""
AI Engine 1 – BioMistral 7B Remote Provider
=============================================
OpenAI-compatible client for BioMistral 7B running in LM Studio
on AI Engine 2 (accessed via Tailscale private network).

This provider is structurally identical to LMStudioProvider but is
explicitly named for BioMistral and targets AI Engine 2's endpoint.
It implements the same LLMProvider interface so the engine's dispatch
logic can route to it via LLM_PROVIDER=biomistral_remote.

Usage:
    provider = BioMistralRemoteProvider()
    if await provider.is_available_async():
        response = await provider.generate("What is hypertension?")

Configuration (read from ai_engine_1.config / .env):
    LLM_PROVIDER=biomistral_remote
    BIOMISTRAL_BASE_URL=http://100.79.169.64:1234
    BIOMISTRAL_MODEL=BioMistral-7B
    BIOMISTRAL_TIMEOUT=120.0
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List

import httpx

from ai_engine_1.llm.provider_base import LLMProvider
from ai_engine_1.llm.llm_engine import LLMResponse
from ai_engine_1.config import settings

try:
    from backend.logging.logger import get_logger
    logger = get_logger("ai-engine-1-biomistral")
except Exception:
    import logging
    logger = logging.getLogger("ai-engine-1-biomistral")


class BioMistralRemoteProvider(LLMProvider):
    """
    Remote BioMistral 7B inference provider.

    Connects to LM Studio running on AI Engine 2 over the Tailscale
    private network.  Uses the same OpenAI-compatible /v1/chat/completions
    endpoint as LMStudioProvider but is configured separately to make
    provider attribution unambiguous in logs and status checks.

    Design notes:
    - Non-streaming (stream=False) — consistent with existing pipeline.
    - Separate timeout from lmstudio_timeout to allow independent tuning.
    - Raises typed exceptions so the engine can distinguish connection
      failures from model errors.
    - Never returns an empty string silently.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (
            base_url
            or getattr(settings, "biomistral_base_url", "http://100.79.169.64:1234")
        ).rstrip("/")
        self.model_id = (
            model_id
            or getattr(settings, "biomistral_model", "BioMistral-7B")
        )
        self.timeout = timeout or float(
            getattr(settings, "biomistral_timeout", 120.0)
        )

        logger.info(
            f"[LLM] BioMistralRemoteProvider initialized | "
            f"base_url={self.base_url} model={self.model_id} timeout={self.timeout}s"
        )

    # ── LLMProvider Interface ─────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """
        Generate a response via BioMistral on AI Engine 2.

        Sends a POST to /v1/chat/completions.  On success, returns an
        LLMResponse.  On failure raises a descriptive exception — never
        returns an empty string silently.

        Error taxonomy:
            ConnectionError  – AI Engine 2 unreachable at Tailscale IP.
            TimeoutError     – Inference did not complete within timeout.
            RuntimeError     – Model returned an empty or malformed response.
        """
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

        timeout_config = httpx.Timeout(
            connect=8.0,    # Tailscale adds negligible latency — generous connect window
            read=self.timeout,
            write=15.0,
            pool=5.0,
        )

        logger.info(
            f"[LLM][BioMistral] Sending inference request | "
            f"endpoint={endpoint} model={self.model_id} max_tokens={max_tokens}"
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
                    raise RuntimeError(
                        f"BioMistral returned empty content. "
                        f"model={self.model_id} response_keys={list(data.keys())}"
                    )

                usage = data.get("usage", {})
                tokens_used = usage.get(
                    "total_tokens", usage.get("completion_tokens", 0)
                )
                latency = (time.time() - start_time) * 1000

                logger.info(
                    f"[LLM][BioMistral] Response received | "
                    f"tokens={tokens_used} latency={round(latency, 1)}ms "
                    f"provider=BioMistral/{self.model_id}"
                )

                return LLMResponse(
                    text=content,
                    model_used=f"BioMistral/{self.model_id}",
                    tokens_used=tokens_used,
                    latency_ms=round(latency, 2),
                    fallback_triggered=False,
                )

            except httpx.ConnectError as conn_err:
                logger.warning(
                    f"[LLM][BioMistral] Connection refused at {self.base_url} "
                    f"(AI Engine 2 unreachable): {conn_err}"
                )
                raise ConnectionError(
                    f"BioMistral service is unreachable at {self.base_url}. "
                    "Verify AI Engine 2 Tailscale connectivity and LM Studio is running."
                ) from conn_err

            except httpx.TimeoutException as time_err:
                logger.warning(
                    f"[LLM][BioMistral] Request timed out after {self.timeout}s: {time_err}"
                )
                raise TimeoutError(
                    f"BioMistral inference timed out after {self.timeout}s. "
                    "Consider increasing BIOMISTRAL_TIMEOUT for long contexts."
                ) from time_err

            except httpx.HTTPStatusError as http_err:
                status_code = http_err.response.status_code
                body = http_err.response.text[:300]
                logger.warning(
                    f"[LLM][BioMistral] HTTP {status_code} from LM Studio: {body}"
                )
                if status_code == 404:
                    raise RuntimeError(
                        f"BioMistral model '{self.model_id}' not found in LM Studio. "
                        "Verify BIOMISTRAL_MODEL matches the model loaded in LM Studio."
                    ) from http_err
                if status_code == 503:
                    raise RuntimeError(
                        "BioMistral model is not loaded yet in LM Studio (503). "
                        "Wait for the model to finish loading."
                    ) from http_err
                raise RuntimeError(
                    f"BioMistral LM Studio returned HTTP {status_code}: {body}"
                ) from http_err

            except Exception as exc:
                logger.warning(f"[LLM][BioMistral] Inference error: {exc}")
                raise

    def is_available(self) -> bool:
        """Synchronous health check — probes /v1/models."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/v1/models")
                return res.status_code == 200
        except Exception:
            return False

    async def is_available_async(self) -> bool:
        """Non-blocking async health check."""
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(f"{self.base_url}/v1/models")
                return res.status_code == 200
        except Exception:
            return False

    async def check_health(self) -> Dict[str, Any]:
        """
        Detailed health probe — returns model list and latency.
        Used by /health/llm endpoint.
        """
        start_t = time.time()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/v1/models")
                latency_ms = round((time.time() - start_t) * 1000, 2)
                if res.status_code == 200:
                    data = res.json()
                    models = [
                        m.get("id") for m in data.get("data", []) if m.get("id")
                    ]
                    return {
                        "status": "online",
                        "provider": "BioMistral",
                        "base_url": self.base_url,
                        "configured_model": self.model_id,
                        "available_models": models,
                        "latency_ms": latency_ms,
                        "error": None,
                    }
                return {
                    "status": "degraded",
                    "provider": "BioMistral",
                    "base_url": self.base_url,
                    "http_status": res.status_code,
                    "error": res.text[:200],
                }
        except Exception as exc:
            return {
                "status": "offline",
                "provider": "BioMistral",
                "base_url": self.base_url,
                "error": str(exc),
            }

    def provider_name(self) -> str:
        return f"BioMistral/{self.model_id}"

    def unload(self) -> None:
        """No local resources — inference is remote."""
        pass
