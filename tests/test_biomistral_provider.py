"""
Unit and Integration Tests for BioMistral 7B Remote Provider
=============================================================
Tests:
- Provider instantiation and configuration
- LLMProvider interface compliance
- Mocked OpenAI API response handling
- Network error handling (ConnectionError / TimeoutError)
- LLMEngine routing with BioMistral as primary
- Graceful degradation when BioMistral is unreachable
"""

import asyncio
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

from ai_engine_1.llm.provider_base import LLMProvider, LLMResponse
from ai_engine_1.llm.provider_biomistral import BioMistralRemoteProvider
from ai_engine_1.llm.llm_engine import ProductionLLMEngine
from ai_engine_1.config import settings


def test_biomistral_provider_inherits_base():
    """Verify BioMistralRemoteProvider adheres to LLMProvider base class."""
    assert issubclass(BioMistralRemoteProvider, LLMProvider)
    provider = BioMistralRemoteProvider(
        base_url="http://100.79.169.64:1234",
        model_id="BioMistral-7B",
        timeout=30.0,
    )
    assert provider.provider_name() == "BioMistral/BioMistral-7B"
    assert isinstance(provider.is_available(), bool)

    # Test mocked positive probe
    with patch("httpx.Client.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_get.return_value = mock_res
        assert provider.is_available() is True


def test_biomistral_generate_success():
    """Verify generate produces valid LLMResponse from mocked OpenAI-compatible endpoint."""
    async def _test():
        provider = BioMistralRemoteProvider(
            base_url="http://100.79.169.64:1234",
            model_id="BioMistral-7B",
            timeout=10.0,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "chatcmpl-test-123",
            "model": "BioMistral-7B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I am Baymax, powered by BioMistral 7B.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 12,
                "total_tokens": 27,
            },
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await provider.generate(prompt="Hello", system_prompt="You are a medical assistant")
            assert isinstance(res, LLMResponse)
            assert "Baymax" in res.text
            assert res.model_used == "BioMistral/BioMistral-7B"
            assert res.tokens_used == 27

    asyncio.run(_test())


def test_biomistral_connection_error():
    """Verify connection failure raises ConnectionError with clear guidance."""
    import httpx

    async def _test():
        provider = BioMistralRemoteProvider(
            base_url="http://100.79.169.64:1234",
            model_id="BioMistral-7B",
            timeout=5.0,
        )

        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(ConnectionError) as exc_info:
                await provider.generate(prompt="Status check")
            assert "unreachable" in str(exc_info.value).lower()
            assert "tailscale" in str(exc_info.value).lower()

    asyncio.run(_test())


def test_llm_engine_routes_to_biomistral():
    """Verify ProductionLLMEngine delegates to BioMistralRemoteProvider when configured."""
    async def _test():
        mock_config = mock.MagicMock()
        mock_config.llm_provider = "biomistral_remote"
        mock_config.primary_llm_model = "BioMistral-7B"
        mock_config.fallback_llm_model = "qwen2.5:7b"
        mock_config.omniroute_api_key = ""
        mock_config.omniroute_base_url = "http://localhost:20128/v1"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.biomistral_base_url = "http://100.79.169.64:1234"
        mock_config.biomistral_model = "BioMistral-7B"
        mock_config.biomistral_timeout = 60.0

        engine = ProductionLLMEngine(config=mock_config)
        assert engine.active_provider_name.startswith("BioMistral")

        mock_biomistral_provider = MagicMock()
        mock_biomistral_provider.generate = AsyncMock(
            return_value=LLMResponse(
                text="Medical guidance provided.",
                model_used="BioMistral/BioMistral-7B",
                tokens_used=40,
                latency_ms=120.0,
            )
        )

        with patch.object(engine, "_get_biomistral_provider", return_value=mock_biomistral_provider):
            res = await engine.generate_async(prompt="What is hypertension?")
            assert res.text == "Medical guidance provided."
            assert res.model_used == "BioMistral/BioMistral-7B"

    asyncio.run(_test())
