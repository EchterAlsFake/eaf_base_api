import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from curl_cffi.requests import Cookies

import base_api.base as base_module
from base_api.base import BaseCore, CachePolicy
from base_api.modules.config import RuntimeConfig
from base_api.modules.errors import (
    ChallengeRegexError,
    NetworkRequestError,
    RequestRetriesExhausted,
)


def response(content=b"ok", status=200, headers=None):
    return SimpleNamespace(
        content=content,
        status_code=status,
        headers=headers or {"content-type": "text/html"},
        encoding="utf-8",
    )


def challenge(token=123):
    return response(
        ('<body onload="go()"><script>function go(){'
         'var p=15; var s=7; n=leastFactor(p);'
         f'document.cookie="KEY="+n+"*"+(p/n)+":"+s+":{token}:1";'
         '}</script></body>').encode()
    )


def core_with_session(request):
    core = BaseCore(RuntimeConfig())
    core.session = SimpleNamespace(headers={}, cookies=Cookies(), request=request)
    return core


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Record retry delays without making a regression take a minute to fail."""
    delays = []
    original_sleep = asyncio.sleep

    async def sleep(delay):
        delays.append(float(delay))
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return delays


@pytest.mark.asyncio
async def test_concurrent_late_challenges_share_one_solution(monkeypatch, recorded_sleeps):
    count = 50
    all_sent = asyncio.Event()
    first_retried = asyncio.Event()
    calls = Counter()
    factor = Mock(wraps=base_module.least_factors)
    monkeypatch.setattr(base_module, "least_factors", factor)

    async def request(*, url, cookies, **kwargs):
        index = int(url.rsplit("/", 1)[1])
        calls[index] += 1
        if calls[index] == 1:
            assert "KEY" not in cookies
            if len(calls) == count:
                all_sent.set()
            await all_sent.wait()
            if index != 0:
                # These old responses arrive AFTER the first solve and replay.
                await first_retried.wait()
            return challenge(123 + index)
        assert cookies["KEY"] == "3*5:7:123:1"
        if index == 0:
            first_retried.set()
        return response(f"video {index}".encode())

    core = core_with_session(request)
    results = await asyncio.wait_for(
        asyncio.gather(*(
            core.fetch_text(f"https://www.pornhub.com/video/{index}",
                            cache_policy=CachePolicy.BYPASS)
            for index in range(count)
        )),
        timeout=5,
    )

    assert results == [f"video {index}" for index in range(count)]
    assert list(calls.values()) == [2] * count
    factor.assert_called_once_with(15)
    assert core.session.cookies.get_dict()["KEY"] == "3*5:7:123:1"
    assert not any(recorded_sleeps)


@pytest.mark.asyncio
async def test_cookie_snapshot_waits_for_solver_lock(recorded_sleeps):
    request = AsyncMock(return_value=response())
    core = core_with_session(request)
    reached_delay = asyncio.Event()

    async def enforce_delay():
        reached_delay.set()

    core.enforce_delay = enforce_delay
    async with core.lock:
        task = asyncio.create_task(core.request("https://www.pornhub.com/video/1"))
        try:
            await asyncio.wait_for(reached_delay.wait(), timeout=5)
            request.assert_not_awaited()
            core.session.cookies.set("KEY", "updated", domain=".pornhub.com", path="/")
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
    await asyncio.wait_for(task, timeout=5)
    assert request.call_args.kwargs["cookies"]["KEY"] == "updated"


@pytest.mark.asyncio
async def test_restored_cookie_is_in_request_snapshot():
    request = AsyncMock(return_value=response())
    core = core_with_session(request)
    core.latest_key = "restored"
    core.latest_key_time = asyncio.get_running_loop().time()

    await core.request("https://www.pornhub.com/video/1")

    assert request.call_args.kwargs["cookies"]["KEY"] == "restored"


@pytest.mark.asyncio
async def test_fresh_challenge_can_be_solved_again(monkeypatch, recorded_sleeps):
    factor = Mock(wraps=base_module.least_factors)
    monkeypatch.setattr(base_module, "least_factors", factor)
    request = AsyncMock(side_effect=[challenge(), challenge(456), response()])
    core = core_with_session(request)

    await core.request("https://www.pornhub.com/video/1")

    assert factor.call_count == 2
    assert request.call_args.kwargs["cookies"]["KEY"] == "3*5:7:456:1"
    assert not any(recorded_sleeps)


@pytest.mark.asyncio
async def test_persistent_challenge_exhausts_retry_budget(recorded_sleeps):
    request = AsyncMock(return_value=challenge())
    core = core_with_session(request)
    core.configuration.request_attempts = 3

    with pytest.raises(RequestRetriesExhausted):
        await core.request("https://www.pornhub.com/video/1")

    assert request.await_count == 3
    assert not core.lock.locked()
    assert not any(recorded_sleeps)


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in", [False, True])
async def test_challenge_respects_non_idempotent_retry_opt_in(opt_in, recorded_sleeps):
    request = AsyncMock(side_effect=[challenge(), response()])
    core = core_with_session(request)
    if opt_in:
        await core.request("https://www.pornhub.com/video/1", method="POST",
                           retry_non_idempotent=True)
    else:
        with pytest.raises(NetworkRequestError):
            await core.request("https://www.pornhub.com/video/1", method="POST")
    assert request.await_count == (2 if opt_in else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    NetworkRequestError("connection reset"),
    response(status=503),
    response(status=429, headers={"Retry-After": "7"}),
])
async def test_network_errors_keep_backoff(failure, recorded_sleeps):
    request = AsyncMock(side_effect=[failure, response()])
    core = core_with_session(request)
    core.configuration.request_retry_initial_delay = 7
    core.configuration.request_retry_jitter = 0

    await core.request("https://www.pornhub.com/video/1")

    assert recorded_sleeps == [7.0]


@pytest.mark.asyncio
async def test_malformed_challenge_releases_lock(recorded_sleeps):
    request = AsyncMock(return_value=response(b'<body onload="go()">invalid</body>'))
    core = core_with_session(request)

    with pytest.raises(ChallengeRegexError):
        await core.request("https://www.pornhub.com/video/1")

    assert request.await_count == 1
    assert not core.lock.locked()
    assert not any(recorded_sleeps)
