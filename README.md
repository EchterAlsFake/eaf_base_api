> [!WARNING]
> Version 4 deliberately removes the legacy `BaseMedia.load(api=..., html=...)`
> and TaskGroup-based `Helper.iterator()` contracts. Applications must migrate
> to source-aware media fields and the new scrape stream described below.

# EAF Base API

# What is this?
When using one of my Porn site APIs, you probably came across this package and wondered what it actually does, so here's
a detailed answer. 

A lot of Porn sites use very similar methods for m3u8 (HLS) parsing and other things. I also wanted to implement proxy
support, and there was a lot of code that I would have rewritten in every API again and again. That's why I made this API
package. The `BaseCore` class does all the necessary stuff like m3u8 parsing, a great caching system, network request
fetching with retry attempts and proxy support.

# Documentation (IMPORTANT!) 
> [!IMPORTANT]
> Configuring eaf_base_api is necessary if you use any of my Porn APIs, because they all depend on this project.
> Please read through the documentation to learn how `PROXIES`, `CACHING` and `LOGGING` etc... work!

You can find the documentation here ->: https://github.com/EchterAlsFake/API_Docs/blob/master/Porn_APIs/eaf_base_api.md

## Source-aware media models

Use `media_field()` for every attribute populated by a remote loader. The first
source is the highest-priority source if multiple sources provide the same field.
Each configured loader is async and returns a complete mapping for all fields
assigned to that source; loaders do not mutate the model directly.

```python
from dataclasses import dataclass
from typing import ClassVar

from base_api import BaseMedia, media_field


@dataclass(kw_only=True, slots=True)
class Video(BaseMedia):
    title: str | None = media_field("html", "api")
    available_qualities: list[int] | None = media_field("html")

    loader_methods: ClassVar[dict[str, str]] = {
        "html": "_load_html",
        "api": "_load_api",
    }

    async def _load_html(self) -> dict[str, object]:
        data = await fetch_and_parse_html(self.url)
        return {
            "title": data.get("title"),
            "available_qualities": data.get("available_qualities"),
        }

    async def _load_api(self) -> dict[str, object]:
        data = await fetch_and_parse_api(self.url)
        return {"title": data.get("title")}
```

Load exactly the information a caller needs:

```python
video = Video(url=url, core=core)
await video.load_fields("title", "available_qualities")

# Or request a known source explicitly.
await video.load_sources("html")

# Convenience form that loads one field and returns it.
title = await video.get_field("title")
```

An unresolved field raises `DataNotLoadedError` with the exact field and eligible
sources. A loader returning `None` marks the field as loaded and does not raise.
Loader mappings are validated before any values are committed, preventing partial
model updates after parser failures.

## Concurrent page and media iteration

`Helper` uses bounded `asyncio` task sets. Completion order is the default because
it exposes fast media without waiting for slower earlier media. Original page and
extractor order is available with `ResultOrder.ORIGINAL`.

```python
from base_api import Helper, ResultOrder

helper = Helper(core=core, constructor=Video)
stream = helper.iterator(
    page_urls,
    extractor_videos,
    max_page_concurrency=3,
    max_item_concurrency=20,
    load_fields=("title", "available_qualities"),
    order=ResultOrder.COMPLETION,  # The default.
)

# The context manager guarantees immediate task cleanup if this loop breaks early.
async with stream:
    async for result in stream:
        if not result.succeeded:
            logger.error("%s failed: %s", result.stage, result.error)
            continue
        video = result.unwrap()
```

Use `order=ResultOrder.ORIGINAL` when presentation order matters. Page and item
failures independently support `ErrorMode.YIELD`, `ErrorMode.SKIP`, or
`ErrorMode.RAISE`. `RetryPolicy` provides a strict maximum attempt count and
optional exponential delay; error handlers return an `ErrorAction` and cannot
create an unbounded retry loop.

# Can I use this for myself?
Yes, you can, but I may change stuff here and there from time to time, and it would maybe break your project.
I would not recommend you to install and use it as a package, but just copy the code you need.

I can recommend everyone the download functions for HLS streaming since, for example, the threaded preset is very well 
optimized. If you just use mine, you need to consume less caffeine and brain cells to make such a function :)

# License
Licensed under The [AGPLv3](https://opensource.org/license/agpl-3-0) license.
<br>Copyright (C) 2024-2026 Johannes Habel
