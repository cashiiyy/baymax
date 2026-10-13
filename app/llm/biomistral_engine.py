"""
BAYMAX AI – BioMistral 7B Remote LLM Engine
=============================================
Replaces QwenEngine as the primary LLM for the app-layer chat pipeline.

Connects to BioMistral 7B running in LM Studio on AI Engine 2 (achu,
100.79.169.64) via the Tailscale private network.

This class mirrors the QwenEngine public interface so app/api/routes/chat.py
requires no structural changes — only the import and instantiation change.

Usage (drop-in replacement for QwenEngine):
    from app.llm.biomistral_engine import BioMistralEngine
    engine = BioMistralEngine()
    response = engine.generate(messages)
    print(response.text)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import httpx

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Result from a single BioMistral generation call.
    Interface-compatible with QwenEngine.LLMResponse.

    Attributes:
        text:           Generated response text.
        input_tokens:   Tokens in the prompt (from LM Studio usage).
        output_tokens:  Tokens generated (from LM Studio usage).
        elapsed_s:      Wall-clock generation time.
        finish_reason:  Why generation stopped.
        provider:       Provider identifier string.
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    finish_reason: str = "eos"
    provider: str = "BioMistral"

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s == 0:
            return 0.0
        return round(self.output_tokens / self.elapsed_s, 2)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class BioMistralEngine:
    """
    Remote inference engine for BioMistral 7B via LM Studio on AI Engine 2.

    Implements the same public interface as QwenEngine:
        engine.generate(messages, stream=False) -> LLMResponse

    Messages must be in the OpenAI chat format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Configuration (reads from app config or environment):
        BIOMISTRAL_BASE_URL  – LM Studio base URL on AI Engine 2
        BIOMISTRAL_MODEL     – Model ID as shown in LM Studio
        BIOMISTRAL_TIMEOUT   – Seconds before inference times out
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        try:
            from config import settings
            self.base_url = (
                base_url
                or getattr(settings, "BIOMISTRAL_BASE_URL", "http://100.79.169.64:1234")
            ).rstrip("/")
            self.model_id = (
                model_id
                or getattr(settings, "BIOMISTRAL_MODEL", "BioMistral-7B")
            )
            self.timeout = timeout or float(
                getattr(settings, "BIOMISTRAL_TIMEOUT", 120.0)
            )
        except Exception:
            self.base_url = (base_url or "http://100.79.169.64:1234").rstrip("/")
            self.model_id = model_id or "BioMistral-7B"
            self.timeout = timeout or 120.0

        log.info(
            "BioMistralEngine initialized | base_url={} model={} timeout={}s",
            self.base_url, self.model_id, self.timeout,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """
        Synchronous generation — calls the async implementation in a thread-safe way.

        Args:
            messages:    OpenAI-format message list.
            stream:      Ignored (streaming not supported in this path).
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.

        Returns:
            LLMResponse with generated text and metadata.

        Raises:
            ConnectionError  if AI Engine 2 is unreachable.
            TimeoutError     if inference does not complete in time.
            RuntimeError     if BioMistral returns empty content.
        """
        import asyncio

        if stream:
            log.warning(
                "BioMistralEngine: streaming requested but not supported in sync mode. "
                "Falling back to non-streaming."
            )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self._generate_async(messages, temperature, max_tokens)
                )
            else:
                return loop.run_until_complete(
                    self._generate_async(messages, temperature, max_tokens)
                )
        except RuntimeError:
            return asyncio.run(
                self._generate_async(messages, temperature, max_tokens)
            )

    async def _generate_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Internal async HTTP call to LM Studio on AI Engine 2."""
        import re
        start_time = time.time()
        endpoint = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        timeout_config = httpx.Timeout(
            connect=8.0,
            read=self.timeout,
            write=15.0,
            pool=5.0,
        )

        log.info(
            "[BioMistral] Sending request | endpoint={} model={} messages={}",
            endpoint, self.model_id, len(messages),
        )

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

                content = ""
                finish_reason = "eos"
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    content = msg.get("content", "").strip()
                    finish_reason = choice.get("finish_reason", "eos")

                if not content:
                    raise RuntimeError(
                        f"BioMistral returned empty content. "
                        f"model={self.model_id} response_keys={list(data.keys())}"
                    )

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                # Auto-recovery for trimmed/cut-off responses ending on dangling intro colon
                if content.endswith(":") or re.search(r"(tips|help|try|following|steps|measures|options):\s*$", content, re.IGNORECASE):
                    log.info("[BioMistral] Detected incomplete list ending with colon. Requesting continuation...")
                    cont_messages = list(messages) + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": "Please continue and provide the complete actionable tips and instructions."}
                    ]
                    cont_payload = {
                        "model": self.model_id,
                        "messages": cont_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    }
                    try:
                        cont_resp = await client.post(endpoint, json=cont_payload)
                        cont_resp.raise_for_status()
                        cont_data = cont_resp.json()
                        if "choices" in cont_data and len(cont_data["choices"]) > 0:
                            cont_text = cont_data["choices"][0].get("message", {}).get("content", "").strip()
                            if cont_text:
                                content = f"{content}\n\n{cont_text}"
                                cont_usage = cont_data.get("usage", {})
                                output_tokens += cont_usage.get("completion_tokens", 0)
                                finish_reason = cont_data["choices"][0].get("finish_reason", "eos")
                    except Exception as cont_err:
                        log.warning("[BioMistral] Auto-continuation failed: {}", cont_err)

                elapsed = time.time() - start_time

                log.info(
                    "[BioMistral] Response received | tokens_in={} tokens_out={} "
                    "elapsed={:.2f}s tok/s={:.1f}",
                    input_tokens, output_tokens, elapsed,
                    output_tokens / elapsed if elapsed > 0 else 0,
                )

                return LLMResponse(
                    text=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    elapsed_s=round(elapsed, 3),
                    finish_reason=finish_reason,
                    provider=f"BioMistral/{self.model_id}",
                )

            except httpx.ConnectError as conn_err:
                log.error(
                    "[BioMistral] Connection refused at {} (AI Engine 2 unreachable): {}",
                    self.base_url, conn_err,
                )
                raise ConnectionError(
                    f"BioMistral service is unreachable at {self.base_url}. "
                    "Check that AI Engine 2 is online and LM Studio is running."
                ) from conn_err

            except httpx.TimeoutException as time_err:
                elapsed = time.time() - start_time
                log.error(
                    "[BioMistral] Inference timed out after {:.1f}s: {}",
                    elapsed, time_err,
                )
                raise TimeoutError(
                    f"BioMistral inference timed out after {self.timeout}s. "
                    "Consider increasing BIOMISTRAL_TIMEOUT."
                ) from time_err

            except httpx.HTTPStatusError as http_err:
                status_code = http_err.response.status_code
                body = http_err.response.text[:300]
                log.error(
                    "[BioMistral] HTTP {} from LM Studio: {}", status_code, body,
                )
                if status_code == 404:
                    raise RuntimeError(
                        f"BioMistral model '{self.model_id}' not found in LM Studio. "
                        "Verify BIOMISTRAL_MODEL matches the loaded model."
                    ) from http_err
                raise RuntimeError(
                    f"BioMistral LM Studio returned HTTP {status_code}: {body}"
                ) from http_err

            except Exception as exc:
                log.error("[BioMistral] Unexpected error: {}", exc)
                raise

    def is_available(self) -> bool:
        """Synchronous health check — probes /v1/models."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/v1/models")
                return res.status_code == 200
        except Exception:
            return False

    def unload(self) -> None:
        """No local resources to release — inference is remote."""
        log.info("BioMistralEngine.unload() called (no-op, model is remote)")
