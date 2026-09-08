import asyncio
import logging
from dataclasses import dataclass
from typing import ClassVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from curl_cffi.requests.cookies import Cookies

from base_api import BaseMedia, BaseCore, media_field
from base_api.modules.config import RuntimeConfig
from base_api.modules.errors import MediaLoadError, NetworkRequestError, RequestRetriesExhausted
from base_api.modules.logger import configure_app_logging


@dataclass(kw_only=True)
class BrokenVideo(BaseMedia):
    url: str
    title: str | None = media_field("html")
    loader_methods: ClassVar = {"html": "_load_html"}

    async def _load_html(self):
        raise ValueError("missing player metadata")


@pytest.mark.asyncio
async def test_direct_media_loading_logs_original_traceback(caplog):
    video = BrokenVideo(url="https://example.test/video/123", core=BaseCore(RuntimeConfig()))
    with caplog.at_level(logging.ERROR), pytest.raises(MediaLoadError) as caught:
        await video.load_sources("html")
    record = next(r for r in caplog.records if "BrokenVideo.html" in r.getMessage())
    assert video.url in record.getMessage()
    assert record.exc_info[1] is caught.value.original_error
    assert record.exc_info[2] is not None
    assert "_load_html" in caplog.text
    assert "ValueError: missing player metadata" in caplog.text


@pytest.mark.asyncio
async def test_media_cancellation_does_not_emit_error(caplog, monkeypatch):
    video = BrokenVideo(url="https://example.test/video/123", core=BaseCore(RuntimeConfig()))
    monkeypatch.setattr(BrokenVideo, "_load_html", AsyncMock(side_effect=asyncio.CancelledError))
    with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
        await video.load_sources("html")
    assert not caplog.records


@pytest.mark.asyncio
async def test_failed_segment_retains_traceback_even_when_returned_as_false(caplog, monkeypatch):
    core = BaseCore(RuntimeConfig())
    original = OSError("connection closed")
    monkeypatch.setattr(core, "fetch_bytes", AsyncMock(side_effect=original))
    url = "https://cdn.example.test/segment-007.ts"
    with caplog.at_level(logging.WARNING, logger=core.logger.name):
        assert await core.download_segment(url, timeout=1) == (url, b"", False)
    assert url in caplog.text
    assert "OSError: connection closed" in caplog.text
    assert any(r.exc_info and r.exc_info[1] is original for r in caplog.records)


def test_configured_log_contains_location_url_and_traceback(tmp_path):
    path = tmp_path / "error.log"
    logger = configure_app_logging("test.error-reporting", log_file=str(path))
    try:
        try:
            raise ValueError("bad response")
        except ValueError:
            logger.exception("Failed to load https://example.test/video/123")
        content = path.read_text()
        assert "test_error_logging.py:" in content
        assert "test_configured_log_contains_location_url_and_traceback" in content
        assert "https://example.test/video/123" in content
        assert "Traceback (most recent call last)" in content
        assert "ValueError: bad response" in content
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


@pytest.mark.asyncio
async def test_exhausted_request_logs_last_error_traceback(caplog, monkeypatch):
    config = RuntimeConfig()
    config.request_attempts = 2
    config.request_retry_initial_delay = 0
    config.request_retry_max_delay = 0
    config.request_retry_jitter = 0
    core = BaseCore(config)
    original = NetworkRequestError("connection reset")
    core.session = SimpleNamespace(
        headers={}, cookies=Cookies(),
        request=AsyncMock(side_effect=original),
    )
    monkeypatch.setattr(core, "initialize_session", lambda: None)
    url = "https://example.test/video/123"
    with caplog.at_level(logging.ERROR), pytest.raises(RequestRetriesExhausted) as caught:
        await core.request(url)
    assert caught.value.__cause__ is original
    record = next(r for r in caplog.records if "after 2 attempts" in r.getMessage())
    assert url in record.getMessage()
    assert record.exc_info[1] is original
    assert record.exc_info[2] is not None
    assert "NetworkRequestError: connection reset" in caplog.text
