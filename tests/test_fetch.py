import asyncio
from collections import deque
from typing import Any, cast

import pytest
import pytest_asyncio
from curl_cffi.requests import AsyncSession

from base_api.base import (
    BaseCore,
    Cache,
    CachePolicy,
    RequestCacheKey,
    SegmentCacheKey,
)
from base_api.modules.config import RuntimeConfig
from base_api.modules.errors import HTTPStatusError, RequestRetriesExhausted


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
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"ok",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "text/plain"}
        self.encoding = "utf-8"


class FakeSession:
    def __init__(self, *outcomes: FakeResponse | Exception, delay: float = 0) -> None:
        self.headers: dict[str, str] = {}
        self.cookies = FakeCookies()
        self.outcomes = deque(outcomes)
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def request(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


@pytest_asyncio.fixture
async def base_core():
    configuration = RuntimeConfig()
    configuration.request_retry_initial_delay = 0
    configuration.request_retry_max_delay = 0
    configuration.request_retry_jitter = 0
    core = BaseCore(configuration=configuration)
    yield core
    await core.close()


def install_session(core: BaseCore, session: FakeSession) -> None:
    core.session = cast(AsyncSession, session)


def request_key(url: str = "https://example.test/cache") -> RequestCacheKey:
    return RequestCacheKey("GET", url, True, "params", "body", "headers", "cookies")


def test_cache_enforces_byte_limits_and_separates_segment_keys():
    configuration = RuntimeConfig()
    configuration.response_cache_size_bytes = 4
    configuration.segment_cache_size_bytes = 100
    cache = Cache(configuration)

    cache.set_response(request_key(), "12345")
    assert cache.get_response(request_key()) is None

    cache.set_response(request_key(), "1234")
    assert cache.get_response(request_key()) == "1234"

    first_key = SegmentCacheKey("https://example.test/master-a", "bc")
    second_key = SegmentCacheKey("https://example.test/master-ab", "c")
    cache.set_segments(first_key, ["one"])
    cache.set_segments(second_key, ["two"])
    first = cache.get_segments(first_key)
    assert first == ["one"]
    assert cache.get_segments(second_key) == ["two"]
    assert first is not None
    first.append("mutation")
    assert cache.get_segments(first_key) == ["one"]


@pytest.mark.asyncio
async def test_request_returns_response(base_core: BaseCore):
    session = FakeSession(FakeResponse(content=b"response"))
    install_session(base_core, session)

    response = await base_core.request("https://example.test/resource")

    assert response.content == b"response"
    assert session.calls[0]["method"] == "GET"


@pytest.mark.asyncio
async def test_fetch_text_and_bytes_have_exact_return_types(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(content="Grüße".encode()),
        FakeResponse(content=b"\x00\x01"),
    )
    install_session(base_core, session)

    text = await base_core.fetch_text("https://example.test/text")
    binary = await base_core.fetch_bytes("https://example.test/binary")

    assert text == "Grüße"
    assert binary == b"\x00\x01"


@pytest.mark.asyncio
async def test_cache_key_includes_params_headers_and_cookies(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(content=b"first"),
        FakeResponse(content=b"second"),
        FakeResponse(content=b"third"),
        FakeResponse(content=b"fourth"),
        FakeResponse(content=b"fifth"),
    )
    install_session(base_core, session)
    url = "https://example.test/items"

    assert await base_core.fetch_text(url, params={"page": 1}) == "first"
    assert await base_core.fetch_text(url, params={"page": 1}) == "first"
    assert await base_core.fetch_text(url, params={"page": 2}) == "second"
    assert await base_core.fetch_text(
        url, params={"page": 2}, headers={"X-View": "full"}
    ) == "third"
    assert await base_core.fetch_text(
        url, params={"page": 2}, cookies={"user": "two"}
    ) == "fourth"
    assert await base_core.fetch_text(
        url, params={"page": 2}, allow_redirects=False
    ) == "fifth"
    assert len(session.calls) == 5


@pytest.mark.asyncio
async def test_cache_policy_bypass_and_refresh(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(content=b"cached"),
        FakeResponse(content=b"bypassed"),
        FakeResponse(content=b"refreshed"),
    )
    install_session(base_core, session)
    url = "https://example.test/volatile"

    assert await base_core.fetch_text(url) == "cached"
    assert await base_core.fetch_text(url, cache_policy=CachePolicy.BYPASS) == "bypassed"
    assert await base_core.fetch_text(url) == "cached"
    assert await base_core.fetch_text(url, cache_policy=CachePolicy.REFRESH) == "refreshed"
    assert await base_core.fetch_text(url) == "refreshed"
    assert len(session.calls) == 3


@pytest.mark.asyncio
async def test_post_is_neither_cached_nor_retried_by_default(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(status_code=500), FakeResponse(content=b"unused")
    )
    install_session(base_core, session)

    with pytest.raises(HTTPStatusError) as caught:
        await base_core.fetch_text(
            "https://example.test/action", method="POST", data={"value": 1}
        )

    assert caught.value.status_code == 500
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_non_idempotent_retry_requires_explicit_opt_in(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(status_code=503), FakeResponse(content=b"recovered")
    )
    install_session(base_core, session)

    result = await base_core.fetch_text(
        "https://example.test/action",
        method="POST",
        data={"value": 1},
        retry_non_idempotent=True,
    )

    assert result == "recovered"
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_retryable_get_status_is_retried(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(status_code=503), FakeResponse(content=b"recovered")
    )
    install_session(base_core, session)

    result = await base_core.fetch_text("https://example.test/unstable")

    assert result == "recovered"
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_retry_after_is_handled_by_retry_policy(base_core: BaseCore):
    session = FakeSession(
        FakeResponse(status_code=429, headers={"Retry-After": "0"}),
        FakeResponse(content=b"recovered"),
    )
    install_session(base_core, session)

    result = await base_core.fetch_text("https://example.test/rate-limited")

    assert result == "recovered"
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_exhausted_retries_preserve_last_error(base_core: BaseCore):
    base_core.configuration.request_attempts = 2
    session = FakeSession(
        FakeResponse(status_code=503), FakeResponse(status_code=503)
    )
    install_session(base_core, session)

    with pytest.raises(RequestRetriesExhausted) as caught:
        await base_core.request("https://example.test/down")

    assert caught.value.attempts == 2
    assert isinstance(caught.value.last_error, HTTPStatusError)
    assert caught.value.last_error.status_code == 503


@pytest.mark.asyncio
async def test_non_retryable_status_fails_once(base_core: BaseCore):
    session = FakeSession(FakeResponse(status_code=404))
    install_session(base_core, session)

    with pytest.raises(HTTPStatusError) as caught:
        await base_core.request("https://example.test/missing")

    assert caught.value.status_code == 404
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_cache_misses_share_one_request(base_core: BaseCore):
    session = FakeSession(FakeResponse(content=b"shared"), delay=0.01)
    install_session(base_core, session)

    results = await asyncio.gather(
        *(base_core.fetch_text("https://example.test/shared") for _ in range(5))
    )

    assert results == ["shared"] * 5
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_context_manager_closes_session():
    configuration = RuntimeConfig()
    core = BaseCore(configuration)
    session = FakeSession()
    install_session(core, session)

    async with core as entered:
        assert entered is core

    assert session.closed is True
    assert core.session is None
