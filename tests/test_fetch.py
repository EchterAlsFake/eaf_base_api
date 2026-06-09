import pytest
import asyncio
from base_api.base import BaseCore
from base_api.modules.config import RuntimeConfig
from base_api.modules.errors import NetworkingError

import pytest_asyncio

@pytest_asyncio.fixture
async def base_core():
    config = RuntimeConfig()
    core = BaseCore(configuration=config)
    core.initialize_session()
    yield core
    if core.session:
        await core.session.close()

@pytest.mark.asyncio
async def test_fetch_get_response(base_core):
    """Test fetch method returning the full response object."""
    url = "https://postman-echo.com/get"
    response = await base_core.fetch(url, get_response=True)
    assert response is not None
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["url"] == url

@pytest.mark.asyncio
async def test_fetch_get_bytes(base_core):
    """Test fetch method returning bytes."""
    url = "https://raw.githubusercontent.com/pytest-dev/pytest/main/LICENSE"
    data = await base_core.fetch(url, get_bytes=True)
    assert isinstance(data, bytes)
    assert len(data) > 10

@pytest.mark.asyncio
async def test_fetch_json(base_core):
    """Test fetch method returning json dict (requires get_json=True if implemented or just getting string and parsing)."""
    # Assuming fetch returns string by default if get_response and get_bytes are False
    url = "https://postman-echo.com/get?foo=bar"
    data = await base_core.fetch(url)
    assert isinstance(data, str)
    assert "foo" in data

@pytest.mark.asyncio
async def test_fetch_404_error(base_core):
    """Test fetch handles 404 errors appropriately."""
    url = "https://postman-echo.com/status/404"
    # By default, fetch might raise an error on 404, or return None. Let's see what it does.
    # We will just assert that it raises an Exception or handles it according to the codebase.
    with pytest.raises(Exception):
        await base_core.fetch(url, error_pass=False)

@pytest.mark.asyncio
async def test_fetch_timeout(base_core):
    """Test fetch timeout mechanism."""
    # httpbin delay endpoint waits for the specified seconds
    url = "https://postman-echo.com/delay/5"
    with pytest.raises(Exception):
        # Using a 1-second timeout, this should fail.
        await base_core.fetch(url, timeout=1)

@pytest.mark.asyncio
async def test_fetch_post_method(base_core):
    """Test fetch with POST method and payload."""
    url = "https://postman-echo.com/post"
    payload = {"key": "value"}
    response = await base_core.fetch(url, method="POST", data=payload, get_response=True)
    assert response.status_code == 200
    json_data = response.json()
    # Assuming data was sent as form data or json, httpbin reflects it in `form` or `json`
    assert json_data.get("form", {}).get("key") == "value" or json_data.get("data", {}).get("key") == "value" or json_data.get("json", {}).get("key") == "value"
