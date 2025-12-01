import pytest
from httpx import AsyncClient, ASGITransport

import wrapper


@pytest.mark.anyio("asyncio")
async def test_models_endpoint_no_auth_returns_200():
    async with AsyncClient(
        transport=ASGITransport(app=wrapper.app),
        base_url="http://test"
    ) as ac:
        response = await ac.get("/v1/models")

    assert response.status_code == 200
