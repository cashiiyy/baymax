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
