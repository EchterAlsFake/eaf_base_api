import os
import time
import pytest
import threading
from base_api.base import BaseCore
from base_api.modules.config import RuntimeConfig
from base_api.modules.errors import DownloadCancelled

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
async def test_legacy_download_multipart(base_core, tmp_path):
    """Test legacy download with allow_multipart=True."""
    # GitHub raw supports range requests
    url = "https://raw.githubusercontent.com/pytest-dev/pytest/main/LICENSE"
    file_path = str(tmp_path / "multipart_download.bin")
    
    success = await base_core.legacy_download(
        path=file_path, 
        url=url, 
        allow_multipart=True,
        chunk_size=100, # tiny chunks to test multipart
        max_workers=2
    )
    
    assert success is True
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 100

@pytest.mark.asyncio
async def test_legacy_download_singlepart(base_core, tmp_path):
    """Test legacy download with allow_multipart=False (fallback to linear)."""
    url = "https://raw.githubusercontent.com/pytest-dev/pytest/main/LICENSE"
    file_path = str(tmp_path / "singlepart_download.bin")
    
    success = await base_core.legacy_download(
        path=file_path, 
        url=url, 
        allow_multipart=False
    )
    
    assert success is True
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 100

@pytest.mark.asyncio
async def test_legacy_download_cancellation(base_core, tmp_path):
    """Test legacy download can be correctly cancelled with a stop_event."""
    url = "https://proof.ovh.net/files/10Mb.dat"  # public fast 10MB test file to allow time for cancellation
    file_path = str(tmp_path / "cancelled_download.bin")
    stop_event = threading.Event()
    
    async def cancel_after_delay():
        await asyncio.sleep(0.5)
        stop_event.set()
        
    import asyncio
    
    # Run download and cancellation concurrently
    task1 = asyncio.create_task(
        base_core.legacy_download(
            path=file_path,
            url=url,
            allow_multipart=True,
            stop_event=stop_event
        )
    )
    task2 = asyncio.create_task(cancel_after_delay())
    
    with pytest.raises(DownloadCancelled):
        await task1
    
    await task2

@pytest.mark.asyncio
async def test_legacy_download_invalid_url(base_core, tmp_path):
    """Test legacy download gracefully fails on bad URLs or non-existent files."""
    url = "https://postman-echo.com/status/404"
    file_path = str(tmp_path / "error_download.bin")
    
    with pytest.raises(Exception):
        await base_core.legacy_download(path=file_path, url=url)
