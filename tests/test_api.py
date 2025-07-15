import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.app.schemas import CustomerData

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    test_data = {
        "customerID": "12345",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        # Add all required fields from your schema
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "churn_probability" in response.json()
    assert 0 <= response.json()["churn_probability"] <= 1
