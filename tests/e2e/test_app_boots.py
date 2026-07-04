"""End-to-end smoke test: the FastAPI wrapper app boots and serves /health.

The app is exercised with a plain TestClient (not the context-manager form),
so the lifespan startup checks — which would contact the live HAWKI backend —
are intentionally not triggered. /health itself needs no network.
"""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


def test_health_endpoint_boots():
    from wrapper import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "initial_models" in data
