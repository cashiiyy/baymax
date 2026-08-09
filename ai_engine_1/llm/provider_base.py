"""
AI Engine 1 – LLM Provider Base
=================================
Abstract interface for pluggable LLM providers.

Existing providers (OpenRouter, Gemini, Ollama) remain in ProductionLLMEngine.
New providers (AirLLM/Qwen) implement this interface and are injected into
the engine's fallback chain.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ai_engine_1.llm.llm_engine import LLMResponse


class LLMProvider(ABC):
    """Abstract base class for LLM inference providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user/assembled prompt text.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (0 = greedy).
            max_tokens: Maximum new tokens to generate.

        Returns:
            LLMResponse with generated text and metadata.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is ready to serve inference."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'QwenAirLLM')."""
        ...

    def unload(self) -> None:
        """Release GPU/RAM resources. Override if the provider holds model state."""
        pass
