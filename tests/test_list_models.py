"""Tests for the GET /v1/models endpoint.

The endpoint returns the available models in OpenAI "list" envelope format,
delegating to format_models_response(hawkiClient.models.list()). These tests
fake the hawki client so no network is required.
"""
import pytest
from httpx import ASGITransport, AsyncClient

import wrapper

REFRESHED_AT = 1_700_000_000


class DummyModels:
    """Stand-in for hawkiClient.models with the attributes the formatter needs."""

    def __init__(self, payload):
        self._payload = payload
        self.refreshed_at = REFRESHED_AT

    def list(self):
        return self._payload


class DummyClient:
    def __init__(self, payload):
        self.models = DummyModels(payload)


async def _get_models(monkeypatch, payload):
    monkeypatch.setattr(wrapper, "hawkiClient", DummyClient(payload), raising=False)
    transport = ASGITransport(app=wrapper.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.get("/v1/models", headers={"Authorization": "Bearer any"})


@pytest.mark.asyncio
async def test_models_dict_payload_returns_openai_list(monkeypatch):
    resp = await _get_models(monkeypatch, {"models": ["m1", "m2"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [entry["id"] for entry in body["data"]]
    assert ids == ["m1", "m2"]
    # the formatter stamps each entry with the models' refresh timestamp + owner
    assert body["data"][0]["created"] == REFRESHED_AT
    assert body["data"][0]["owned_by"] == wrapper.DEFAULT_OWNER


@pytest.mark.asyncio
async def test_models_plain_list_payload(monkeypatch):
    resp = await _get_models(monkeypatch, ["only-model"])
    assert resp.status_code == 200
    assert [e["id"] for e in resp.json()["data"]] == ["only-model"]


@pytest.mark.asyncio
async def test_models_endpoint_reports_errors_as_500(monkeypatch):
    class Boom:
        def list(self):
            raise RuntimeError("backend exploded")

    class BoomClient:
        models = Boom()

    monkeypatch.setattr(wrapper, "hawkiClient", BoomClient(), raising=False)
    transport = ASGITransport(app=wrapper.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/models", headers={"Authorization": "Bearer any"})
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "RuntimeError"
