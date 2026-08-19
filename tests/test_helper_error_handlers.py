from collections.abc import Mapping
from typing import Any

import pytest

from base_api.base import BaseCore, Helper
from base_api.modules.config import IteratorConfig, RuntimeConfig
from base_api.modules.errors import ItemFetchError, PageFetchError
from base_api.modules.type_hints import (
    ErrorAction,
    ErrorMode,
    RetryPolicy,
    ScrapeErrorContext,
    ScrapeStage,
)


def _single_item(_: Any) -> list[Mapping[str, Any]]:
    return [{"url": "https://example.test/item"}]


@pytest.mark.asyncio
async def test_item_failures_use_only_the_item_error_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = BaseCore(RuntimeConfig())

    async def fetch_text(*_: Any, **__: Any) -> str:
        return "page"

    monkeypatch.setattr(core, "fetch_text", fetch_text)
    page_contexts: list[ScrapeErrorContext] = []
    item_contexts: list[ScrapeErrorContext] = []

    def page_handler(context: ScrapeErrorContext) -> ErrorAction:
        page_contexts.append(context)
        return ErrorAction.RAISE

    async def item_handler(context: ScrapeErrorContext) -> ErrorAction:
        item_contexts.append(context)
        return ErrorAction.YIELD

    helper: Helper[Any] = Helper(core, lambda **_: object())
    stream = helper.iterator(
        ["https://example.test/page"],
        _single_item,
        iterator_config=IteratorConfig(
            extract_in_thread=False,
            page_retry=RetryPolicy(max_attempts=1),
            item_retry=RetryPolicy(max_attempts=1),
            page_error_mode=ErrorMode.RAISE,
            item_error_mode=ErrorMode.RAISE,
            page_error_handler=page_handler,
            item_error_handler=item_handler,
        ),
    )

    results = [result async for result in stream]

    assert page_contexts == []
    assert len(item_contexts) == 1
    context = item_contexts[0]
    assert context.stage is ScrapeStage.ITEM
    assert context.url == "https://example.test/item"
    assert context.page_index == 0
    assert context.item_index == 0
    assert context.attempt == 1
    assert len(results) == 1
    assert results[0].stage is ScrapeStage.ITEM
    assert isinstance(results[0].error, ItemFetchError)


@pytest.mark.asyncio
async def test_page_failures_use_only_the_page_error_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = BaseCore(RuntimeConfig())

    async def fetch_text(*_: Any, **__: Any) -> str:
        raise LookupError("page unavailable")

    monkeypatch.setattr(core, "fetch_text", fetch_text)
    page_contexts: list[ScrapeErrorContext] = []
    item_contexts: list[ScrapeErrorContext] = []

    def page_handler(context: ScrapeErrorContext) -> ErrorAction:
        page_contexts.append(context)
        return ErrorAction.YIELD

    def item_handler(context: ScrapeErrorContext) -> ErrorAction:
        item_contexts.append(context)
        return ErrorAction.RAISE

    helper: Helper[Any] = Helper(core, lambda **_: object())
    stream = helper.iterator(
        ["https://example.test/page"],
        _single_item,
        iterator_config=IteratorConfig(
            extract_in_thread=False,
            page_retry=RetryPolicy(max_attempts=1),
            item_retry=RetryPolicy(max_attempts=1),
            page_error_mode=ErrorMode.RAISE,
            item_error_mode=ErrorMode.RAISE,
            page_error_handler=page_handler,
            item_error_handler=item_handler,
        ),
    )

    results = [result async for result in stream]

    assert item_contexts == []
    assert len(page_contexts) == 1
    context = page_contexts[0]
    assert context.stage is ScrapeStage.PAGE
    assert context.url == "https://example.test/page"
    assert context.page_index == 0
    assert context.item_index is None
    assert context.attempt == 1
    assert len(results) == 1
    assert results[0].stage is ScrapeStage.PAGE
    assert isinstance(results[0].error, PageFetchError)
