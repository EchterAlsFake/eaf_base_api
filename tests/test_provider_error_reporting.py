"""Offline compatibility checks against provider checkouts beside eaf_base_api.

Run with an environment containing the providers' dependencies. Providers whose
checkout is absent are skipped, so this also works in a standalone base checkout.
"""
import asyncio
import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from base_api.modules import errors
from base_api.modules.config import DownloadConfigHLS, DownloadConfigRAW


PROVIDERS = (
    "beeg", "eporner", "hqporner", "missav", "pornhub", "porntrex",
    "redtube", "spankbang", "thumbzilla", "tube8", "xfreehd", "xhamster",
    "xnxx", "xvideos", "youporn",
)
URL = "https://example.test/video/123"


@pytest.fixture(params=PROVIDERS)
def provider(request, monkeypatch):
    name = request.param
    checkout = Path(__file__).resolve().parents[2] / f"unofficial-api-for-{name}"
    if not checkout.is_dir():
        pytest.skip(f"No sibling checkout for {name}")
    pytest.importorskip("selectolax")
    if name in ("pornhub", "redtube", "xhamster"):
        pytest.importorskip("chompjs")
    if name == "porntrex":
        pytest.importorskip("json5")
    monkeypatch.syspath_prepend(str(checkout))
    return importlib.import_module(f"{name}_api.{'beeg_api' if name == 'beeg' else 'api'}")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type, public_type", [
    (errors.NetworkRequestError, errors.NetworkError),
    (errors.InvalidProxy, errors.ProxyError),
    (errors.BotProtectionDetected, errors.BotDetection),
    (errors.UnknownError, errors.UnknownNetworkError),
])
async def test_request_logs_url_traceback_and_chains_shared_error(provider, caplog, error_type, public_type):
    original = error_type("specific request failure")
    core = SimpleNamespace(fetch_text=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(public_type) as caught:
        await provider.get_html_content(core=core, url=URL)
    assert caught.value.__cause__ is original
    assert URL in str(caught.value)
    record = next(r for r in caplog.records if r.name == provider.logger.name)
    assert URL in record.getMessage()
    assert record.exc_info[1] is original
    assert "Traceback (most recent call last)" in caplog.text
    assert "specific request failure" in caplog.text
    assert "get_html_content" in caplog.text


@pytest.mark.asyncio
async def test_unexpected_request_failure_is_logged_and_preserved(provider, caplog):
    original = ValueError("malformed response")
    core = SimpleNamespace(fetch_text=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError) as caught:
        await provider.get_html_content(core=core, url=URL)
    assert caught.value is original
    assert URL in caplog.text
    assert "ValueError: malformed response" in caplog.text
    assert any(r.exc_info and r.exc_info[2] for r in caplog.records)


@pytest.mark.asyncio
async def test_not_found_remains_detectable(provider, caplog):
    original = errors.HTTPStatusError("missing video", 404, URL)
    core = SimpleNamespace(fetch_text=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(Exception) as caught:
        await provider.get_html_content(core=core, url=URL)
    assert errors.is_resource_gone(caught.value)
    assert caught.value.__cause__ is original
    assert URL in caplog.text
    assert any(r.exc_info and r.exc_info[1] is original for r in caplog.records)


async def download(provider, core, *, load_error=None, use_hls=True):
    media = SimpleNamespace(
        core=core, url=URL, title="Example", load_fields=AsyncMock(side_effect=load_error),
        m3u8_base_url="https://cdn.example.test/master.m3u8", is_hls=use_hls,
        direct_download_urls=["cdn.example.test/720.mp4"],
        cdn_urls=["https://cdn.example.test/720.mp4"], video_qualities=["720"],
        get_url_by_quality=lambda **_: "https://cdn.example.test/720.mp4",
    )
    name = provider.__name__.split("_")[0]
    raw = DownloadConfigRAW(path="unused.mp4", quality="best", no_title=True)
    hls = DownloadConfigHLS(path="unused.mp4", quality="best", no_title=True)
    if name == "spankbang":
        return await provider.Video.download(media, configuration_hls=hls, configuration_raw=raw, use_hls=use_hls)
    if name == "youporn":
        return await provider.Video.download(media, configuration=hls, backup_configuration=raw)
    if name == "eporner":
        return await provider.Video.download(media, configuration=raw, mode="mp4")
    config = raw if name in ("hqporner", "porntrex", "xfreehd") else hls
    return await provider.Video.download(media, configuration=config)


@pytest.mark.asyncio
async def test_download_failure_is_raised_with_cause_and_url(provider, caplog):
    original = OSError("disk full")
    core = SimpleNamespace(download=AsyncMock(side_effect=original), legacy_download=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(errors.DownloadFailed) as caught:
        await download(provider, core)
    assert caught.value.__cause__ is original
    assert URL in str(caught.value)
    assert URL in caplog.text
    assert "OSError: disk full" in caplog.text
    assert any(r.exc_info and r.exc_info[1] is original for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_type", [errors.DownloadCancelled, asyncio.CancelledError])
async def test_download_cancellation_is_not_wrapped_as_failure(provider, caplog, cancel_type):
    original = cancel_type("cancelled by caller")
    core = SimpleNamespace(download=AsyncMock(side_effect=original), legacy_download=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(cancel_type) as caught:
        await download(provider, core)
    assert caught.value is original
    assert not caplog.records


def test_provider_errors_keep_local_and_shared_base_types(provider):
    assert issubclass(provider.DownloadFailed, errors.DownloadFailed)
    if provider.__name__ == "pornhub_api.api":
        local = importlib.import_module("pornhub_api.modules.errors")
        assert isinstance(local.DownloadFailed("failure"), local.PornhubAPIError)
        assert isinstance(local.NotFound("missing"), errors.NotFound)


@pytest.mark.asyncio
async def test_download_preparation_error_keeps_url_and_traceback(provider, caplog):
    original = ValueError("invalid media metadata")
    core = SimpleNamespace(download=AsyncMock(), legacy_download=AsyncMock())
    with caplog.at_level(logging.ERROR), pytest.raises(errors.DownloadFailed) as caught:
        await download(provider, core, load_error=original)
    assert caught.value.__cause__ is original
    assert URL in caplog.text
    assert "ValueError: invalid media metadata" in caplog.text
    core.download.assert_not_awaited()
    core.legacy_download.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["spankbang", "youporn"], indirect=True)
async def test_raw_fallback_download_logs_url_and_cause(provider, caplog):
    original = OSError("raw stream interrupted")
    core = SimpleNamespace(download=AsyncMock(), legacy_download=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(errors.DownloadFailed) as caught:
        await download(provider, core, use_hls=False)
    assert caught.value.__cause__ is original
    assert URL in caplog.text
    assert "OSError: raw stream interrupted" in caplog.text
    core.legacy_download.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_server_error_uses_shared_network_error(provider, caplog):
    original = errors.HTTPStatusError("server unavailable", 503, URL)
    core = SimpleNamespace(fetch_text=AsyncMock(side_effect=original))
    with caplog.at_level(logging.ERROR), pytest.raises(errors.NetworkError) as caught:
        await provider.get_html_content(core=core, url=URL)
    assert caught.value.__cause__ is original
    assert URL in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["xnxx"], indirect=True)
async def test_xnxx_server_error_is_not_reported_as_region_block(provider):
    original = errors.HTTPStatusError("server unavailable", 503, URL)
    core = SimpleNamespace(fetch_text=AsyncMock(side_effect=original))
    with pytest.raises(errors.NetworkError) as caught:
        await provider.get_html_content(core=core, url=URL)
    assert caught.value.__cause__ is original


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["pornhub"], indirect=True)
async def test_stored_scrape_error_logs_original_traceback_outside_except(provider, caplog):
    try:
        raise ValueError("stored extraction failure")
    except ValueError as original:
        error = errors.ItemFetchError(URL, original, 1, 0, 0)
        error.__cause__ = original

    async def results():
        yield SimpleNamespace(succeeded=False, url=URL, error=error)

    with caplog.at_level(logging.ERROR):
        await provider._cli_download_video_generator(
            results(), SimpleNamespace(), True, {"limit": None, "downloaded": 0},
        )
    assert URL in caplog.text
    assert "ValueError: stored extraction failure" in caplog.text
    assert "test_stored_scrape_error_logs_original_traceback_outside_except" in caplog.text
    assert "NoneType: None" not in caplog.text


@pytest.mark.parametrize("provider", ["hqporner", "spankbang"], indirect=True)
def test_custom_errors_have_messages_and_shared_base(provider):
    local = importlib.import_module(f"{provider.__package__}.modules.errors")
    if provider.__package__ == "hqporner_api":
        exceptions = (local.InvalidActress(), local.NotAvailable())
    else:
        exceptions = (local.VideoIsProcessing(),)
    for error in exceptions:
        assert str(error)
        assert isinstance(error, errors.ScraperException)
        assert isinstance(error, errors.BaseScraperError)
