"""Offline URL contracts shared by the provider adapters."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from base_api import BaseCore, append_url_path, profile_base_url, update_url_query
from base_api.modules.config import RuntimeConfig
from base_api.modules.provider import fetch_content
from base_api.modules.urls import FetchedText


def test_query_and_path_pagination_preserve_filters_and_slug_encoding():
    url = "https://de.example.com/channel/a+b/?q=hd&page=7#videos"
    assert update_url_query(url, page=2) == "https://de.example.com/channel/a+b/?q=hd&page=2"
    assert append_url_path(url, 2) == "https://de.example.com/channel/a+b/2?q=hd&page=7"
    assert profile_base_url(url + "/ignored") == "https://de.example.com/channel/a+b?q=hd"


@pytest.mark.asyncio
async def test_redirect_identity_survives_the_text_cache():
    core = BaseCore(RuntimeConfig())
    original = "https://www.example.com/model/name"
    final = "https://www.example.com/pornstar/name"
    core.request = AsyncMock(return_value=SimpleNamespace(content=b"page", encoding="utf-8", url=final))
    async with core:
        for _ in range(2):
            owner = SimpleNamespace(url=original, resolved_url=None)
            result = await fetch_content(core, original, logger=logging.getLogger(__name__), owner=owner)
            assert result == "page"
            assert owner.url == original
            assert owner.resolved_url == final
    core.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_auxiliary_and_foreign_responses_do_not_replace_media_identity():
    original = "https://www.example.com/model/name"
    owner = SimpleNamespace(url=original, resolved_url=None)
    core = SimpleNamespace(fetch_text=AsyncMock(return_value=FetchedText("page", url="https://other.test/login")))
    await fetch_content(core, original, logger=logging.getLogger(__name__), owner=owner)
    assert owner.resolved_url is None
