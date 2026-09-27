import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_web_interface():
    response = client.get("/")
    assert response.status_code == 200
    assert "B.A.Y.M.A.X. v2" in response.text

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "B.A.Y.M.A.X. v2" in data["system"]
    import os
    assert data["backend_ip"] == os.getenv("TAILSCALE_IP", "100.89.251.123")

def test_chat_endpoint():
    response = client.post("/chat", json={"user_id": 1, "query": "What are the symptoms of fever?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "Disclaimer" in data["response"]

def test_history_endpoint():
    response = client.get("/history/1")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert len(data["history"]) > 0


def test_tts_voices_endpoint():
    response = client.get("/tts/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert "voices" in data
    assert any(v.get("name", "").lower() == "baymax" for v in data["voices"])


def test_tts_endpoint_voicebox():
    from unittest.mock import patch, AsyncMock
    with patch("app.tts.voicebox_engine.voicebox_engine.generate_speech", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "success": True,
            "audio_base64": "UklGRg==",
            "duration_seconds": 2.1,
            "voice_used": "Voicebox-Baymax",
            "engine": "voicebox",
            "format": "wav",
        }
        response = client.post("/tts", json={"text": "Hello, I am Baymax.", "voice": "Baymax"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["voice_used"] == "Voicebox-Baymax"
        assert data["audio_base64"] == "UklGRg=="


def test_chat_respond_alias_endpoint():
    response = client.post("/chat/respond", json={"user_id": "1", "text": "What is CPR?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0


def test_dedicated_health_endpoints():
    # RAG health
    r_rag = client.get("/health/rag")
    assert r_rag.status_code == 200
    data_rag = r_rag.json()
    assert "status" in data_rag

    # LLM health
    r_llm = client.get("/health/llm")
    assert r_llm.status_code == 200
    data_llm = r_llm.json()
    assert "status" in data_llm

    # Voice health
    r_voice = client.get("/health/voice")
    assert r_voice.status_code == 200
    data_voice = r_voice.json()
    assert "status" in data_voice

