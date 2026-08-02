import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from ai_engine_1.planner.planner import IntelligentQueryPlanner
from ai_engine_1.safety.safety_engine import MedicalSafetyEngine
from ai_engine_1.explainability.explainer import ExplainabilityEngine
from ai_engine_1.tools.medical_tools import tool_registry
from ai_engine_1.embeddings.embedder import AdvancedEmbedder

client = TestClient(app)

def test_query_planner():
    planner = IntelligentQueryPlanner()
    plan_emergency = planner.plan("Patient has severe chest pain and cannot breathe")
    assert plan_emergency.emergency_protocol is True
    assert plan_emergency.intent == "emergency"

    plan_bmi = planner.plan("Calculate BMI for 70kg and 175cm height")
    assert plan_bmi.tool_required is True

def test_safety_engine():
    safety = MedicalSafetyEngine()
    report = safety.validate_safety("Is aspirin safe during pregnancy?", "Consult doctor before taking medications.")
    assert report.pregnancy_warning is True

def test_medical_tools():
    bmi_res = tool_registry.execute_tool("bmi_calculator", weight_kg=70, height_cm=175)
    assert bmi_res.status == "success"
    assert bmi_res.result["bmi"] == 22.9

    unit_res = tool_registry.execute_tool("unit_converter", value=102, from_unit="Fahrenheit", to_unit="Celsius")
    assert unit_res.result["converted"] == "38.9 Celsius"

def test_embedder_caching():
    embedder = AdvancedEmbedder()
    v1 = embedder.encode("Fever symptoms")
    v2 = embedder.encode("Fever symptoms")
    assert len(v1) == 384
    assert (v1 == v2).all()

def test_engine1_routes():
    res_health = client.get("/engine1/health")
    assert res_health.status_code == 200

    res_status = client.get("/engine1/status")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "ready"

    res_plan = client.post("/engine1/plan", json={"query": "How to handle burns?"})
    assert res_plan.status_code == 200
    assert "intent" in res_plan.json()

    res_embed = client.post("/engine1/embed", json={"text": "Test embedding"})
    assert res_embed.status_code == 200
    assert len(res_embed.json()["vector"]) == 384

    res_reason = client.post("/engine1/reason", json={"query": "What are the first aid steps for burns?"})
    assert res_reason.status_code == 200
    data = res_reason.json()
    assert "response" in data
    assert "confidence" in data
