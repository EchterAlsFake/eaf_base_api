# Migrating an API package to eaf_base_api 4

Version 4 intentionally has no compatibility layer for the old `BaseMedia` flags
or TaskGroup-based Helper implementation. This guide uses the structure of
`unofficial-api-for-pornhub` as the concrete example.

## HTTP request API

The multi-mode `BaseCore.fetch()` method was removed. Replace it according to the
representation the caller needs:

| Old call | Replacement |
| --- | --- |
| `fetch(url)` | `fetch_text(url)` |
| `fetch(url, get_bytes=True)` | `fetch_bytes(url)` |
| `fetch(url, get_response=True)` | `request(url)` |
| `fetch(url, save_cache=False)` | `fetch_text(url, cache_policy=CachePolicy.BYPASS)` |

Use `CachePolicy.REFRESH` to force a network request and store its result. POST and
PATCH requests are not cached or retried by default; pass
`retry_non_idempotent=True` only when replaying the operation is safe. Prefer
`async with BaseCore() as core` or call `await core.close()` explicitly.

The request/cache configuration names now describe their units and semantics:

| Removed setting | Replacement |
| --- | --- |
| `max_cache_items` | `response_cache_size_bytes` and `segment_cache_size_bytes` |
| `max_retries` | `request_attempts` |

TTL and retry timing are configured with `response_cache_ttl`,
`segment_cache_ttl`, `request_retry_initial_delay`,
`request_retry_max_delay`, and `request_retry_jitter`.

## 1. Update imports

Import the field declaration, scheduling, retry, and error-policy types used by
your API:

```python
from base_api import (
    BaseCore,
    BaseMedia,
    ErrorAction,
    ErrorMode,
    Helper,
    ResultOrder,
    RetryPolicy,
    ScrapeErrorContext,
    ScrapeResult,
    ScrapeStream,
    media_field,
)
```

`on_error_hint` was removed. Type handlers as a callable receiving
`ScrapeErrorContext` and returning `ErrorAction`, or import the `ErrorHandler`
type alias from `base_api`.

## 2. Declare the source of every remotely populated field

Replace optional dataclass defaults used as "not loaded" markers:

```python
title: str | None = None
```

with source-aware fields:

```python
title: str | None = media_field("html", "api")
```

The first source has the highest precedence. In this example HTML wins after both
HTML and API have loaded, even if the API request finishes last. A field with only
one source is simpler:

```python
available_qualities: list[int] | None = media_field("html")
```

Passing a value during construction, including `None`, marks it as already
resolved. Omitting it leaves the private unloaded sentinel in place. Application
code never imports or handles that sentinel.

For the current Pornhub models, these source declarations are appropriate:

- `UserHelper`, `Album`, `Short`, `GIF`, `Channel`, and `Playlist`: all remotely
  parsed fields use `media_field("html")`.
- `Video` HTML-only fields: `is_vr`, `is_video_unavailable`, `is_hd`,
  `available_qualities`, `is_vertical`,
  `is_video_unavailable_in_your_country`, `m3u8_base_url`, `author_thumbnail`,
  `author_link`, and `author_information`.
- `Video` fields returned by HTML and API: `duration`, `title`, `thumbnail`,
  `views`, `publish_date`, `likes`, `categories`, and `tags`. Use
  `media_field("html", "api")` if the richer HTML representation should win.
- The API extractor currently returns `rating_percent` without declaring the
  field. Either declare it with `media_field("api")` or remove it from the parser.
- `video_id` is derived from the URL, not fetched remotely. Keep it as a normal
  dataclass field and populate it in `__post_init__`, or expose it as a property.

Normalize fields such as `categories` and `tags` to one stable type. The current
Video HTML parser returns dictionaries while its API parser returns lists. Source
precedence is deterministic, but a public attribute changing type according to
the selected source remains difficult for API users. Also correct the existing
`tags` annotations on Album, GIF, and Playlist: their parsers currently return
dictionaries even though the dataclasses annotate lists.

## 3. Replace `_perform_load` and mutating `_fetch_*` methods

Delete `_perform_load`. Each model maps a source name to an async loader method:

```python
from typing import ClassVar


@dataclass(kw_only=True, slots=True)
class GIF(BaseMedia):
    title: str | None = media_field("html")
    content_url: str | None = media_field("html")

    loader_methods: ClassVar[dict[str, str]] = {
        "html": "_load_html",
    }

    async def _load_html(self) -> dict[str, object]:
        html = await get_html_content(core=self.core, url=self.url)
        if not isinstance(html, str):
            raise TypeError("GIF HTML response must be text")
        return await asyncio.to_thread(self._extract_html, html)
```

Do not call `setattr` in a source loader. Return a mapping and let `BaseMedia`
validate and commit it atomically.

The mapping contract is deliberately strict:

- It must contain every field assigned to that source.
- It must not contain fields that are undeclared or assigned to another source.
- Use an explicit `None` value when the remote service omitted an optional field.
- A parser exception or invalid mapping commits nothing.

Consequently, remove this pattern from every `_fetch_html`/`_fetch_api` method:

```python
allowed_fields = {field.name for field in fields(self)}
for key, value in data.items():
    if key in allowed_fields:
        setattr(self, key, value)
```

That pattern hid parser typos and caused partial state. The returned mapping is now
the parser contract and mistakes raise `LoaderContractError` immediately.

For Video, configure both sources:

```python
loader_methods: ClassVar[dict[str, str]] = {
    "html": "_load_html",
    "api": "_load_api",
}
```

The old GIF edge case inside `Video._perform_load` returned a different object,
but `BaseMedia.load` always returned the original Video. Select `GIF` versus
`Video` before construction (ideally in the page extractor or API factory) rather
than trying to change model type from inside a loader.

## 4. Replace media loading calls

Old calls:

```python
await video.load(api=True, html=False)
await album.load(html=True)
```

become explicit source or field requests:

```python
await video.load_sources("api")
await album.load_sources("html")

# Prefer this when the caller needs only selected information.
await video.load_fields("title", "duration")

# Load one field and return it.
title = await video.get_field("title")
```

`load_fields` chooses a small set of sources covering all requested unresolved
fields. Repeated requests are idempotent, and concurrent requests for the same
source share a task.

Client factory methods should construct, await, and return the concrete object:

```python
async def get_video(self, url: str, *, sources: tuple[str, ...] = ("api",)) -> Video:
    video = Video(url=url, core=self.core)
    await video.load_sources(*sources)
    return video
```

The loading methods return `Self`, so `return await video.load_sources("api")` is
also correctly typed.

Model methods should load their own prerequisites instead of assuming callers
already selected the correct source. For example:

```python
async def download(self, configuration: DownloadConfigRAW) -> bool:
    await self.load_fields("title", "content_url")
    ...

async def get_author(self) -> Pornstar:
    author_link = await self.get_field("author_link")
    author = Pornstar(url=author_link, core=self.core)
    await author.load_fields("bio", "about")
    return author
```

This is the intended form of dynamic loading: ordinary attribute access stays
synchronous and gives a precise `DataNotLoadedError`, while an async method uses
`load_fields`/`get_field` immediately before it needs remote data. Search API
packages for both old `.load(` calls and older `.init()` calls; the latter still
appear in Video's author helper and should become explicit field/source loads.

## 5. Replace Helper arguments and result access

The principal argument changes are:

| Removed name | Version 4 replacement |
|---|---|
| `video_link_extractor` | `item_extractor` |
| `max_video_concurrency` | `max_item_concurrency` |
| `fetch_html`, `fetch_api`, `fetch_anything_else` | `load_sources` or `load_fields` |
| `keep_original_order` | `order=ResultOrder.ORIGINAL` |
| `ignore_errors` | `page_error_mode` and `item_error_mode` |
| boolean retry callbacks | bounded `RetryPolicy` plus an `ErrorAction` handler |

Completion order is the default:

```python
stream = helper.iterator(
    target_page_urls=page_urls,
    item_extractor=extractor_videos,
    max_page_concurrency=pages_concurrency,
    max_item_concurrency=videos_concurrency,
    load_sources=("api",),
    order=ResultOrder.COMPLETION,
)
```

To reproduce page order followed by each extractor's item order:

```python
order=ResultOrder.ORIGINAL
```

`ScrapeResult.video` and the mutable `is_success` flag no longer exist. Results
are immutable and generic:

```python
if result.succeeded:
    video = result.unwrap()       # Or use result.item after checking.
else:
    logger.error("%s failed: %s", result.stage, result.error)
```

Page errors can now be yielded too, so check `result.stage` when the API exposes
them. If an API should expose media failures but never page-failure results, use
`page_error_mode=ErrorMode.RAISE` or `ErrorMode.SKIP`.

## 6. Own stream cleanup

Use the stream as an async context manager. This is essential when a consumer may
break early, because it immediately cancels outstanding page fetches and media
loads:

```python
stream = helper.iterator(...)
async with stream:
    async for result in stream:
        yield result
```

Where an API method only builds URLs and has no setup awaits, consider returning
`ScrapeStream` directly instead of wrapping it in another async generator:

```python
def search_videos(...) -> ScrapeStream[Video]:
    return self.helper.iterator(...)
```

The API caller can then deterministically own cleanup:

```python
async with client.search_videos(...) as videos:
    async for result in videos:
        ...
```

If an API keeps an async-generator wrapper, its own callers must close that outer
generator when breaking early; the inner `async with` cleans Helper as soon as the
outer generator is closed.

## 7. Replace retry callbacks

Old handlers returned a boolean and could retry forever. New handlers receive all
context and return an explicit action:

```python
async def on_item_error(context: ScrapeErrorContext) -> ErrorAction:
    if isinstance(context.error, ResourceGone):
        return ErrorAction.SKIP
    return ErrorAction.RETRY
```

Pair the handler with a hard attempt limit:

```python
stream = helper.iterator(
    ...,
    item_retry=RetryPolicy(
        max_attempts=4,
        base_delay=0.5,
        multiplier=2.0,
        max_delay=8.0,
    ),
    item_error_handler=on_item_error,
    item_error_mode=ErrorMode.YIELD,
)
```

Returning `RETRY` on the final allowed attempt falls back to `item_error_mode`.
Handler exceptions are always fatal `ErrorHandlerError` instances; they are not
mistaken for remote item failures.

## 8. Empty pages and extractor requirements

Helper no longer treats every empty page as a global pagination sentinel. An
empty extractor result represents a successful page containing zero items, and
the remaining explicit target URLs are still processed. This avoids a faster
higher-numbered page incorrectly pruning valid lower-numbered work.

Extractors must be synchronous and return an iterable of mappings. Helper runs the
complete extraction iteration in a worker thread by default. Each mapping must
contain a non-empty string `url` (or the configured `item_url_key`) and must be
accepted by the media constructor along with `core`.

## 9. Recommended Pornhub migration sequence

1. Convert one HTML-only model, such as GIF, to `media_field("html")` and a pure
   `_load_html` mapping loader.
2. Convert all remaining HTML-only models using the same pattern.
3. Convert Video, explicitly resolving overlapping field types and HTML/API
   precedence.
4. Change direct client factory methods from `.load(...)` to `load_sources` or
   `load_fields`.
5. Change each Helper call and result consumer, using completion order by default.
6. Replace boolean callbacks with bounded policies and `ErrorAction` handlers.
7. Add mocked tests for loader mapping completeness, yielded page/item failures,
   both order modes, retry exhaustion, and early stream closure before restoring
   live integration tests.
