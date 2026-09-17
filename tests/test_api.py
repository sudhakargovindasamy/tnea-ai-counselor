"""
tests/test_api.py
FastAPI Endpoints and Integration Test Suite for TNEA Counselor System.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "TNEA Counselor AI" in response.text

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "features" in data
    assert "l1_cache" in data["features"]
    assert "sse_streaming" in data["features"]

def test_search_colleges_district_and_branch(client):
    response = client.get("/search_colleges?district=Coimbatore&branch_code=CS&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 3
    for college in data["results"]:
        assert college["district"] == "Coimbatore"
        assert "CS" in college["branch_codes"]

def test_search_colleges_hostel(client):
    response = client.get("/search_colleges?district=Coimbatore&has_hostel=true&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 3
    for college in data["results"]:
        assert college["district"] == "Coimbatore"
        assert college.get("hostel_facilities_boys") or college.get("hostel_facilities_girls") or college.get("mess_bill_boys")

def test_clear_chat(client):
    session_id = "test-session-pytest-123"
    response = client.post(f"/clear_chat/{session_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"

def test_cutoff_query_graceful_handling(client):
    payload = {
        "question": "Which college can I get with a cutoff of 185 in Coimbatore?",
        "session_id": "pytest-cutoff-session",
        "top_k": 5
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "cutoff" in data["answer"].lower() or "tneaonline.org" in data["answer"].lower()
    # Sources should be empty because vector search was skipped
    assert data.get("sources") == []

def test_chat_sse_cutoff_stream(client):
    payload = {
        "question": "Which college can I get with a cutoff of 185 in Coimbatore?",
        "session_id": "pytest-sse-session"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "data:" in content
    assert "tneaonline.org" in content or "cutoff" in content
    assert "data: [DONE]" in content

def test_admin_purge_cache_unauthorized(client):
    response = client.post("/admin/purge_cache?admin_secret=wrong_secret_key")
    assert response.status_code == 403

def test_admin_purge_cache_authorized(client):
    valid_secret = os.getenv("ADMIN_SECRET_KEY", "TNEA_SUPER_SECRET_ADMIN_KEY_2026")
    response = client.post(f"/admin/purge_cache?admin_secret={valid_secret}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

