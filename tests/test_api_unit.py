"""Unit tests for FastAPI endpoints using TestClient."""

from fastapi.testclient import TestClient
import pytest

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health reports healthy status and model load state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_valid():
    """Verify /predict returns valid positive duration prediction."""
    payload = {
        "PULocationID": "43",
        "DOLocationID": "151",
        "trip_distance": 2.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "duration_minutes" in data
    assert data["duration_minutes"] > 0


def test_predict_endpoint_negative_distance():
    """Verify /predict rejects negative distance with 400 Bad Request."""
    payload = {
        "PULocationID": "43",
        "DOLocationID": "151",
        "trip_distance": -2.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_endpoint_missing_field():
    """Verify /predict rejects payloads missing required fields with 422."""
    payload = {
        "PULocationID": "43",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
