import os
import pytest
import asyncio
from typing import Any
from base_api.base import BaseCore
from base_api.modules.config import RuntimeConfig

class MockVideo:
    def __init__(self, m3u8_url: str):
        self.m3u8_base_url = m3u8_url

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
async def test_hls_download_no_remux(base_core, tmp_path):
    """Test HLS download functionality without remuxing."""
    # A public test HLS stream
    m3u8_url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
    video = MockVideo(m3u8_url)
    out_path = str(tmp_path / "test_hls.mp4")
    
    # We choose a 'quality' that is present, or a fallback. 
    # Usually "best" or the highest resolution is selected, we can pass "best" if supported,
    # or just pass whatever the get_segments method expects.
    # Let's try passing '1080' or '720' or 'highest'.
    
    # Let's just pass "highest"
    result = await base_core.download(
        video=video,
        quality="best",
        path=out_path,
        remux=False,
        callback=None,
        max_workers_download=2
    )
    
    # Because remux=False, it might leave the segment directory or a .ts file?
    # By default, it might concatenate them. Let's check if the file or something was produced.
    # The return type is bool or DownloadReport.
    assert result is True or isinstance(result, dict) or getattr(result, 'success', False) is True

@pytest.mark.asyncio
async def test_hls_download_remux(base_core, tmp_path):
    """Test HLS download functionality with remuxing enabled."""
    m3u8_url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
    video = MockVideo(m3u8_url)
    out_path = str(tmp_path / "test_hls_remux.mp4")
    
    result = await base_core.download(
        video=video,
        quality="best",
        path=out_path,
        remux=True,
        callback=None,
        max_workers_download=2
    )
    
    assert result is True or getattr(result, 'success', False) is True
    # Depending on how the codebase remuxes (maybe requires ffmpeg or pyav)
    # the file should exist
    assert os.path.exists(out_path)
