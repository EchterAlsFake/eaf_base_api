"""Offline regressions for failures at the shared request/download boundaries."""
import asyncio
import logging
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from base_api import BaseCore, BaseMedia, DownloadConfigHLS, DownloadConfigRAW, media_field
from base_api.modules.config import RuntimeConfig
from base_api.modules.errors import DownloadCancelled, DownloadFailed, MediaLoadError
from base_api.modules.logger import get_logger
from base_api.modules.provider import prepare_download_config


@dataclass(kw_only=True)
class ThreadedVideo(BaseMedia):
    title: str | None = media_field("html")
    loader_methods: ClassVar = {"html": "_load_html"}

    async def _load_html(self):
        # Yield so concurrent loaders overlap before entering a worker thread.
        await asyncio.sleep(0)
        await asyncio.to_thread(get_logger(__name__).warning, "Worker parser warning")
        raise ValueError(f"Invalid metadata for {self.url}")


@pytest.mark.asyncio
async def test_concurrent_loaders_keep_class_url_and_thread_context(caplog):
    urls = ["https://one.test/video", "https://two.test/video"]
    videos = [ThreadedVideo(core=object(), url=url) for url in urls]
    with caplog.at_level(logging.WARNING):
        failures = await asyncio.gather(*(v.load_sources("html") for v in videos), return_exceptions=True)
        get_logger(__name__).warning("Outside the loaders")
    assert all(isinstance(error, MediaLoadError) for error in failures)
    records = [r for r in caplog.records if r.getMessage().endswith("Worker parser warning")]
    assert {r.url for r in records} == set(urls)
    assert all(r.class_name.endswith(".ThreadedVideo") for r in records)
    assert caplog.records[-1].url == "-"
    assert caplog.records[-1].class_name == "-"
    assert "ValueError: Invalid metadata" in caplog.text


def test_constructing_core_does_not_change_application_logging():
    logger = logging.getLogger("base_api.base")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    try:
        logger.setLevel(logging.DEBUG)
        BaseCore(RuntimeConfig())
        assert logger.level == logging.DEBUG
        assert logger.handlers == original_handlers
    finally:
        logger.setLevel(original_level)


@pytest.mark.asyncio
async def test_prepared_download_still_observes_callers_stop_signal():
    stop = asyncio.Event()
    configuration = DownloadConfigHLS(quality="best", path="out", stop_event=stop)
    prepared = prepare_download_config(configuration, "Title")
    assert configuration.path == "out"
    assert prepared.path == "out/Title.mp4"
    stop.set()
    assert prepared.stop_event.is_set()


@pytest.mark.asyncio
async def test_hls_error_is_raised_with_original_cause(tmp_path, monkeypatch, caplog):
    core = BaseCore(RuntimeConfig())
    original = OSError("playlist connection closed")
    monkeypatch.setattr(core, "get_segments", AsyncMock(side_effect=original))
    url = "https://cdn.test/master.m3u8"
    config = DownloadConfigHLS(quality="best", path=str(tmp_path / "out.mp4"), m3u8_base_url=url)
    with caplog.at_level(logging.ERROR), pytest.raises(DownloadFailed) as caught:
        await core.download(config)
    assert caught.value.__cause__ is original
    assert caught.value.url == url
    assert "OSError: playlist connection closed" in caplog.text
    assert any(r.url == url and r.class_name == "base_api.base.BaseCore" for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("return_report", [False, True])
async def test_incomplete_hls_download_and_explicit_report(tmp_path, monkeypatch, return_report):
    core = BaseCore(RuntimeConfig())
    url = "https://cdn.test/master.m3u8"
    segment = "https://cdn.test/segment.ts"
    monkeypatch.setattr(core, "get_segments", AsyncMock(return_value=[segment]))
    monkeypatch.setattr(core, "download_segment", AsyncMock(return_value=(segment, b"", False)))
    monkeypatch.setattr("base_api.base.asyncio.sleep", AsyncMock())
    config = DownloadConfigHLS(quality="best", path=str(tmp_path / "out.mp4"), m3u8_base_url=url,
                               return_report=return_report)
    if return_report:
        report = await core.download(config)
        assert report.status == "failed"
        assert report.missing_urls == [segment]
    else:
        with pytest.raises(DownloadFailed, match="segments missing"):
            await core.download(config)
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.asyncio
async def test_raw_download_failure_logs_and_chains_cause(monkeypatch, caplog):
    core = BaseCore(RuntimeConfig())
    original = OSError("disk full")
    monkeypatch.setattr(core, "_legacy_download", AsyncMock(side_effect=original))
    url = "https://cdn.test/video.mp4"
    with caplog.at_level(logging.ERROR), pytest.raises(DownloadFailed) as caught:
        await core.legacy_download(url, DownloadConfigRAW(quality="best"))
    assert caught.value.__cause__ is original
    assert caught.value.url == url
    assert "OSError: disk full" in caplog.text
    assert any(r.url == url for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("return_report", [False, True])
async def test_hls_stop_is_not_logged_as_failure(tmp_path, monkeypatch, caplog, return_report):
    core = BaseCore(RuntimeConfig())
    monkeypatch.setattr(core, "get_segments", AsyncMock(return_value=["https://cdn.test/segment.ts"]))
    stop = asyncio.Event()
    stop.set()
    config = DownloadConfigHLS(quality="best", path=str(tmp_path / "out.mp4"), m3u8_base_url="https://cdn.test/master.m3u8",
                               stop_event=stop, return_report=return_report)
    with caplog.at_level(logging.ERROR):
        if return_report:
            assert (await core.download(config)).status == "cancelled"
        else:
            with pytest.raises(DownloadCancelled):
                await core.download(config)
    assert not caplog.records
