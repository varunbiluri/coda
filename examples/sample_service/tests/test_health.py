"""Tests for the sample service - initially failing to demonstrate the flow."""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint - this will initially fail."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
