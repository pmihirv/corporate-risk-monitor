import pytest
from fastapi.testclient import TestClient
from app import app

# Initialize the simulated test client targeting our FastAPI instance
client = TestClient(app)

def test_dashboard_loading():
    """Verify that the main dashboard UI renders successfully (HTTP 200)"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Enterprise Insolvency Risk Monitor" in response.content

def test_evaluation_pipeline_stable():
    """Verify that a highly solvent profile yields a stable verdict status"""
    payload = {
        "working_capital": 0.35,
        "retained_earnings": 0.50,
        "ebitda": 0.15,
        "sentiment_score": 0.80
    }
    response = client.post("/evaluate", data=payload)
    assert response.status_code == 200
    assert b"STABLE / SOLID SOLVENCY MARGINS" in response.content

def test_evaluation_pipeline_distressed():
    """Verify that a heavily distressed profile correctly flags a critical alert"""
    payload = {
        "working_capital": -0.15,
        "retained_earnings": -0.25,
        "ebitda": -0.05,
        "sentiment_score": -0.85
    }
    response = client.post("/evaluate", data=payload)
    assert response.status_code == 200
    assert b"CRITICAL INSOLVENCY PROFILE DETECTED" in response.content
