from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_endpoint():
    """Verify that the home gateway loads up with the correct metadata framework."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Modified Altman Z-Score (5 Ratios)" in data["framework"]

def test_predict_healthy_company():
    """Test a highly liquid, profitable corporate profile."""
    payload = {
        "working_capital_ta": 0.45,
        "retained_earnings_ta": 0.55,
        "ebitda_ta": 0.35,
        "market_cap_tl": 2.5,
        "sales_ta": 2.1,
        "sentiment": 0.8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "altman_z_score" in data
    assert "financial_risk_zone" in data
    assert "ml_risk_probability" in data
    assert "model_classification" in data

def test_predict_distressed_company():
    """Test a heavily leveraged, deeply distressed corporate profile."""
    payload = {
        "working_capital_ta": -0.15,
        "retained_earnings_ta": -0.40,
        "ebitda_ta": -0.10,
        "market_cap_tl": 0.2,
        "sales_ta": 0.6,
        "sentiment": -0.7
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # The updated model should catch this and calculate a low Z-score zone
    assert data["financial_risk_zone"] == "Distress Zone"