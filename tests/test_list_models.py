import pytest
from httpx import AsyncClient, ASGITransport

import wrapper

class DummyModels:
    def list(self):
        return {"models": ["m1"]}

class DummyClient:
    def __init__(self):
        self.models = DummyModels()

@pytest.mark.asyncio
async def test_api_key_in_allowed_keys_always_allowed(monkeypatch):
    # api_key in ALLOWED_KEYS, is_api_key_working = False -> allowed
    monkeypatch.setattr(wrapper, "ALLOWED_KEYS", ["allowed_key"], raising=False)
    monkeypatch.setattr(wrapper, "is_api_key_working", lambda k: False, raising=False)
    monkeypatch.setattr(wrapper, "hawkiClient", DummyClient(), raising=False)

    async with AsyncClient(transport=ASGITransport(app=wrapper.app), base_url="http://test") as ac:
        headers = {"Authorization": "Bearer allowed_key"}
        r = await ac.get("/v1/models", headers=headers)

    assert r.status_code == 200
    assert r.json() == {"models": ["m1"]}

@pytest.mark.asyncio
async def test_api_key_in_allowed_keys_and_working_true(monkeypatch):
    # api_key in ALLOWED_KEYS, is_api_key_working = True -> allowed
    monkeypatch.setattr(wrapper, "ALLOWED_KEYS", ["allowed_key2"], raising=False)
    monkeypatch.setattr(wrapper, "is_api_key_working", lambda k: True, raising=False)
    monkeypatch.setattr(wrapper, "hawkiClient", DummyClient(), raising=False)

    async with AsyncClient(transport=ASGITransport(app=wrapper.app), base_url="http://test") as ac:
        headers = {"Authorization": "Bearer allowed_key2"}
        r = await ac.get("/v1/models", headers=headers)

    assert r.status_code == 200
    assert r.json() == {"models": ["m1"]}

@pytest.mark.asyncio
async def test_api_key_not_allowed_but_is_working_allowed(monkeypatch):
    # api_key not in ALLOWED_KEYS, is_api_key_working = True -> allowed
    monkeypatch.setattr(wrapper, "ALLOWED_KEYS", [], raising=False)
    monkeypatch.setattr(wrapper, "is_api_key_working", lambda k: True, raising=False)
    monkeypatch.setattr(wrapper, "hawkiClient", DummyClient(), raising=False)

    async with AsyncClient(transport=ASGITransport(app=wrapper.app), base_url="http://test") as ac:
        headers = {"Authorization": "Bearer user_provided_key"}
        r = await ac.get("/v1/models", headers=headers)

    assert r.status_code == 200
    assert r.json() == {"models": ["m1"]}

@pytest.mark.asyncio
async def test_api_key_not_allowed_and_not_working_unauthorized(monkeypatch):
    # api_key not in ALLOWED_KEYS, is_api_key_working = False -> 401
    monkeypatch.setattr(wrapper, "ALLOWED_KEYS", [], raising=False)
    monkeypatch.setattr(wrapper, "is_api_key_working", lambda k: False, raising=False)
    monkeypatch.setattr(wrapper, "hawkiClient", DummyClient(), raising=False)

    async with AsyncClient(transport=ASGITransport(app=wrapper.app), base_url="http://test") as ac:
        headers = {"Authorization": "Bearer bad_key"}
        r = await ac.get("/v1/models", headers=headers)

    assert r.status_code == 401
    assert r.json().get("error") == "Unauthorized"
