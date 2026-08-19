from collections import deque
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from curl_cffi.requests import AsyncSession

import base_api.base as base_module
from base_api.base import BaseCore
from base_api.modules.config import DownloadConfigHLS, RuntimeConfig


class FakeCookies:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get_dict(self) -> dict[str, str]:
        return dict(self._values)

    def set(self, name: str, value: str, **_: Any) -> None:
        self._values[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._values


class FakeResponse:
    status_code = 200
    content = b"ok"
    headers = {"content-type": "text/plain"}
    encoding = "utf-8"


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.headers: dict[str, str] = {}
        self.cookies = FakeCookies()
        self.responses = deque(responses)

    async def request(self, **_: Any) -> FakeResponse:
        return self.responses.popleft()

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_initialize_session_applies_runtime_cookies_and_is_idempotent() -> None:
    configuration = RuntimeConfig()
    configuration.cookies = {"configured": "initial"}
    core = BaseCore(configuration)

    try:
        core.initialize_session()
        session = core.session
        assert session is not None
        assert session.cookies.get_dict()["configured"] == "initial"

        session.cookies.set("dynamic", "preserved")
        configuration.cookies = {"configured": "changed"}
        core.initialize_session()

        assert core.session is session
        assert session.cookies.get_dict()["configured"] == "initial"
        assert session.cookies.get_dict()["dynamic"] == "preserved"
    finally:
        await core.close()


@pytest.mark.asyncio
async def test_close_then_initialize_reapplies_current_runtime_cookies() -> None:
    configuration = RuntimeConfig()
    configuration.cookies = {"configured": "first"}
    core = BaseCore(configuration)

    core.initialize_session()
    first_session = core.session
    await core.close()

    configuration.cookies = {"configured": "second"}
    try:
        core.initialize_session()
        assert core.session is not None
        assert core.session is not first_session
        assert core.session.cookies.get_dict()["configured"] == "second"
    finally:
        await core.close()


def test_initialize_session_preserves_an_injected_session() -> None:
    core = BaseCore(RuntimeConfig())
    injected = FakeSession()
    core.session = cast(AsyncSession, injected)

    core.initialize_session()

    assert core.session is injected


@pytest.mark.asyncio
async def test_hls_download_uses_the_cores_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = RuntimeConfig()
    configuration.timeout = 7
    configuration.max_workers_download = 2
    core = BaseCore(configuration)
    threaded_download = AsyncMock(return_value=True)
    monkeypatch.setattr(core, "threaded_download", threaded_download)
    download_configuration = DownloadConfigHLS(
        quality="best",
        m3u8_base_url="https://example.test/master.m3u8",
    )

    result = await core.download(download_configuration)

    assert result is True
    threaded_download.assert_awaited_once_with(
        configuration=download_configuration,
        pre_resolved_m3u8="https://example.test/master.m3u8",
        timeout=7,
        max_workers=2,
    )


@pytest.mark.asyncio
async def test_request_backoff_uses_the_runtime_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = RuntimeConfig()
    configuration.request_multiplier = 3.5
    core = BaseCore(configuration)
    core.session = cast(AsyncSession, FakeSession(FakeResponse()))
    captured: dict[str, float] = {}

    def fake_wait_exponential_jitter(**kwargs: float) -> Any:
        captured.update(kwargs)
        return lambda _: 0.0

    monkeypatch.setattr(
        base_module,
        "wait_exponential_jitter",
        fake_wait_exponential_jitter,
    )

    await core.request("https://example.test/resource")

    assert captured["exp_base"] == 3.5
