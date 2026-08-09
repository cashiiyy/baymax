"""
Tests for the Qwen 2.5 Local Provider + Observation Schema upgrade.

Tests cover:
1. Provider base class contract
2. QwenLocalProvider is_available() graceful handling
3. Config loading with new env vars
4. Fallback chain (local unavailable → OpenRouter → Gemini → Offline)
5. Observation schema validation
6. Deterministic risk assessment
7. Observation prompt building
8. Structured output parsing
9. Malformed output handling (no crash)
10. Existing API endpoints still work
"""

import pytest
import json
from unittest.mock import patch, MagicMock

# ── 1. Provider Base Class ────────────────────────────────────────────────────

def test_provider_base_is_abstract():
    """LLMProvider cannot be instantiated directly."""
    from ai_engine_1.llm.provider_base import LLMProvider
    with pytest.raises(TypeError):
        LLMProvider()


# ── 2. QwenLocalProvider Availability ─────────────────────────────────────────

def test_qwen_local_provider_availability_check():
    """QwenLocalProvider.is_available() should return True/False without crashing."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    provider = QwenLocalProvider(model_id="Qwen/Qwen2.5-7B-Instruct")
    result = provider.is_available()
    assert isinstance(result, bool)


def test_qwen_local_provider_name():
    """provider_name() returns a descriptive string."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    provider = QwenLocalProvider(model_id="Qwen/Qwen2.5-7B-Instruct")
    name = provider.provider_name()
    assert "QwenLocal" in name
    assert "Qwen2.5" in name


def test_qwen_local_provider_unload_safe():
    """unload() should not crash even when no model is loaded."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    provider = QwenLocalProvider()
    provider.unload()  # Should not raise


# ── 3. Config Loading ─────────────────────────────────────────────────────────

def test_config_has_qwen_fields():
    """EngineConfig should have the new Qwen local inference fields."""
    from ai_engine_1.config import EngineConfig
    config = EngineConfig()
    assert hasattr(config, 'llm_provider')
    assert hasattr(config, 'qwen_model')
    assert hasattr(config, 'qwen_max_new_tokens')
    assert hasattr(config, 'qwen_temperature')
    assert hasattr(config, 'qwen_top_p')
    assert hasattr(config, 'qwen_repetition_penalty')


def test_config_defaults():
    """EngineConfig should default to omniroute."""
    from ai_engine_1.config import EngineConfig
    config = EngineConfig()
    assert config.llm_provider == "omniroute"
    assert config.qwen_model == "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    assert config.qwen_max_new_tokens == 512


# ── 4. Fallback Chain ─────────────────────────────────────────────────────────

def test_llm_engine_default_provider_type():
    """Default ProductionLLMEngine should use omniroute provider type."""
    from ai_engine_1.llm.llm_engine import ProductionLLMEngine
    engine = ProductionLLMEngine()
    assert engine._llm_provider_type == "omniroute"


def test_llm_engine_active_provider_name():
    """active_provider_name should return a meaningful string."""
    from ai_engine_1.llm.llm_engine import ProductionLLMEngine
    engine = ProductionLLMEngine()
    name = engine.active_provider_name
    assert isinstance(name, str)
    assert len(name) > 0


def test_llm_engine_offline_fallback():
    """With no API keys and no local model, should fall back to offline."""
    from ai_engine_1.config import EngineConfig
    config = EngineConfig(omniroute_api_key="")  # No API key
    from ai_engine_1.llm.llm_engine import ProductionLLMEngine
    engine = ProductionLLMEngine(config=config)

    # This should NOT crash — it should reach the offline fallback
    response = engine.generate("What are the symptoms of fever?")
    assert response.text  # Should have some offline response
    assert response.fallback_triggered is True


# ── 5. Observation Schema ─────────────────────────────────────────────────────

def test_observation_schema_valid():
    """VisionObservation should accept valid data."""
    from ai_engine_1.pipeline.observation_schema import VisionObservation
    obs = VisionObservation(
        event_type="possible_fall",
        confidence=0.91,
        duration_seconds=8.2,
        movement_state="minimal",
        facial_state="distressed",
        person_detected=True,
    )
    assert obs.event_type == "possible_fall"
    assert obs.confidence == 0.91
    assert obs.person_detected is True


def test_observation_schema_defaults():
    """VisionObservation should have sensible defaults."""
    from ai_engine_1.pipeline.observation_schema import VisionObservation
    obs = VisionObservation(event_type="person_detected", confidence=0.85)
    assert obs.movement_state == "unknown"
    assert obs.facial_state == "unknown"
    assert obs.person_detected is False
    assert obs.timestamp  # Should have auto-generated timestamp


def test_observation_schema_rejects_invalid_confidence():
    """Confidence must be between 0 and 1."""
    from ai_engine_1.pipeline.observation_schema import VisionObservation
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        VisionObservation(event_type="test", confidence=1.5)


# ── 6. Deterministic Risk Assessment ─────────────────────────────────────────

def test_risk_assessment_critical_fall():
    """High-confidence fall should be classified as critical."""
    from ai_engine_1.pipeline.observation_schema import (
        VisionObservation, assess_observation_risk
    )
    obs = VisionObservation(
        event_type="possible_fall",
        confidence=0.92,
        duration_seconds=5.0,
        movement_state="none",
        facial_state="distressed",
        person_detected=True,
    )
    risk = assess_observation_risk(obs)
    assert risk.risk_level == "critical"
    assert risk.requires_immediate_action is True
    assert risk.escalation_recommended is True
    assert len(risk.safety_triggers) > 0


def test_risk_assessment_low_confidence_fall():
    """Low-confidence fall should be high risk, not critical."""
    from ai_engine_1.pipeline.observation_schema import (
        VisionObservation, assess_observation_risk
    )
    obs = VisionObservation(
        event_type="possible_fall",
        confidence=0.5,
        movement_state="minimal",
    )
    risk = assess_observation_risk(obs)
    assert risk.risk_level == "high"
    assert risk.requires_immediate_action is False


def test_risk_assessment_normal_activity():
    """Normal person detection should be low risk."""
    from ai_engine_1.pipeline.observation_schema import (
        VisionObservation, assess_observation_risk
    )
    obs = VisionObservation(
        event_type="person_detected",
        confidence=0.95,
        movement_state="active",
        facial_state="neutral",
        person_detected=True,
    )
    risk = assess_observation_risk(obs)
    assert risk.risk_level == "low"
    assert risk.requires_immediate_action is False


def test_risk_assessment_prolonged_inactivity():
    """Long inactivity with no movement should escalate."""
    from ai_engine_1.pipeline.observation_schema import (
        VisionObservation, assess_observation_risk
    )
    obs = VisionObservation(
        event_type="person_detected",
        confidence=0.88,
        duration_seconds=60.0,
        movement_state="none",
        person_detected=True,
    )
    risk = assess_observation_risk(obs)
    assert risk.risk_level == "high"
    assert risk.escalation_recommended is True


# ── 7. Observation Prompt Building ────────────────────────────────────────────

def test_observation_prompt_has_sections():
    """Built prompt should contain fact/inference separated sections."""
    from ai_engine_1.pipeline.observation_schema import (
        VisionObservation, assess_observation_risk, build_observation_prompt
    )
    obs = VisionObservation(
        event_type="possible_fall",
        confidence=0.91,
        movement_state="minimal",
        facial_state="distressed",
        person_detected=True,
    )
    risk = assess_observation_risk(obs)
    prompt = build_observation_prompt(obs, risk)

    assert "OBSERVED FACTS" in prompt
    assert "SAFETY ASSESSMENT" in prompt
    assert "YOUR TASK" in prompt
    assert "possible_fall" in prompt


# ── 8. Structured Output Parsing ──────────────────────────────────────────────

def test_parse_structured_json():
    """Should parse valid JSON from model output."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    raw = '{"situation": "possible_fall", "severity": "high"}'
    result = QwenLocalProvider.parse_structured_response(raw)
    assert result is not None
    assert result["situation"] == "possible_fall"


def test_parse_structured_markdown_json():
    """Should extract JSON from markdown code block."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    raw = 'Here is my analysis:\n```json\n{"severity": "high"}\n```'
    result = QwenLocalProvider.parse_structured_response(raw)
    assert result is not None
    assert result["severity"] == "high"


def test_parse_structured_embedded_json():
    """Should find JSON embedded in prose."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    raw = 'The analysis shows: {"severity": "low", "confidence": 0.9} which is good.'
    result = QwenLocalProvider.parse_structured_response(raw)
    assert result is not None
    assert result["severity"] == "low"


# ── 9. Malformed Output Handling ──────────────────────────────────────────────

def test_parse_structured_invalid_returns_none():
    """Malformed output should return None, not crash."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    result = QwenLocalProvider.parse_structured_response("This is just plain text.")
    assert result is None


def test_parse_structured_empty_returns_none():
    """Empty string should return None, not crash."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    result = QwenLocalProvider.parse_structured_response("")
    assert result is None


def test_parse_structured_broken_json_returns_none():
    """Broken JSON should return None, not crash."""
    from ai_engine_1.llm.provider_qwen import QwenLocalProvider
    result = QwenLocalProvider.parse_structured_response('{"severity": "high", broken}')
    assert result is None


# ── 10. Existing Components Still Work ────────────────────────────────────────

def test_planner_still_works():
    """Query planner should still function correctly."""
    from ai_engine_1.planner.planner import IntelligentQueryPlanner
    planner = IntelligentQueryPlanner()
    plan = planner.plan("Patient has severe chest pain and cannot breathe")
    assert plan.emergency_protocol is True
    assert plan.intent == "emergency"


def test_safety_engine_still_works():
    """Safety engine should still function correctly."""
    from ai_engine_1.safety.safety_engine import MedicalSafetyEngine
    safety = MedicalSafetyEngine()
    report = safety.validate_safety("Is aspirin safe during pregnancy?", "Consult doctor.")
    assert report.pregnancy_warning is True


def test_medical_tools_still_work():
    """Medical tools should still function correctly."""
    from ai_engine_1.tools.medical_tools import tool_registry
    bmi_res = tool_registry.execute_tool("bmi_calculator", weight_kg=70, height_cm=175)
    assert bmi_res.status == "success"
    assert bmi_res.result["bmi"] == 22.9


def test_embedder_still_works():
    """Embedder should still produce 384-dim vectors."""
    from ai_engine_1.embeddings.embedder import AdvancedEmbedder
    embedder = AdvancedEmbedder()
    v = embedder.encode("Test text")
    assert len(v) == 384
