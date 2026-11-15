"""Validated, lazy pagination using the existing Helper request/item pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import asyncio
from hashlib import sha256
from html.parser import HTMLParser
from inspect import isawaitable
from typing import Callable, Literal
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit
from base_api.modules.urls import same_site_url

from base_api.base import Helper, ScrapeStream
from base_api.modules.config import make_iterator_config
from base_api.modules.type_hints import ErrorAction, ErrorMode

PageCount = int | Literal["all"]


def validate_pagination(pages: PageCount, start_page: int = 1) -> None:
    if isinstance(pages, bool) or (
        pages != "all" and (not isinstance(pages, int) or pages < 1)
    ):
        raise ValueError("pages must be a positive integer or 'all'")
    if (
        isinstance(start_page, bool)
        or not isinstance(start_page, int)
        or start_page < 1
    ):
        raise ValueError("start_page must be a positive integer")


@dataclass(frozen=True)
class PageInfo:
    """Provider-confirmed navigation. None means the terminal state is unknown."""

    terminal: bool | None = None
    next_url: str | None = None
    total_pages: int | None = None
    current_page: int | None = None
    next_is_probe: bool = False


@dataclass
class PaginationSummary:
    status: str = "pending"
    pages: int = 0
    last_page: int | None = None
    errors: int = 0
    duplicates: int = 0

    @property
    def exhausted(self) -> bool:
        return self.status == "exhausted"


def _identity(url):
    p = urlsplit(url)
    return (
        p.scheme,
        p.netloc,
        p.path.rstrip("/"),
        tuple(sorted(parse_qsl(p.query, keep_blank_values=True))),
    )


def _page_signature(rows):
    if not rows:
        return None
    digest = sha256()
    for row in rows:
        value = str(row.get("url") or row.get("video_id") or row).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.digest()


def _check_listing_response(content, requested):
    resolved = getattr(content, "url", None)
    if not isinstance(resolved, str):
        return
    before = urlsplit(requested).path.rstrip("/")
    after = urlsplit(resolved).path.rstrip("/")
    if before and not after:
        raise ValueError("Listing redirected to the site homepage")
    if (
        after in {"/login", "/signin", "/register", "/registerstep1", "/user/login"}
        and after != before
    ):
        raise ValueError("Listing redirected to an account access page")


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.next_hrefs = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").casefold().split()
        next_marker = "next" in attrs.get("rel", "").casefold().split() or any(
            token in {"next", "next-page", "next_page", "nmnext", "pagination-next"}
            or token.endswith(("--next", "-next", "_next"))
            for token in classes
        )
        inherited = next_marker or any(n for _, n in self.stack)
        if tag == "a" and attrs.get("href"):
            self.hrefs.append(attrs["href"])
            if inherited:
                self.next_hrefs.append(attrs["href"])
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append((tag, inherited))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


def linked_page(
    content, page: int, rows, *, page_url: Callable[[int], str]
) -> PageInfo:
    """For HTML endpoints whose numbered anchors describe the complete navigation.

    Match the actual next request rather than translated link labels or broad
    class-name matches that can accidentally select video cards/recommendations.
    Providers with AJAX blocks or unreliable navigation must supply another parser.
    """
    parser = _Links()
    parser.feed(str(content))
    expected = page_url(page + 1)
    base = getattr(content, "url", None) or page_url(page)
    if same_site_url(expected, base):
        parts = urlsplit(expected)
        actual = urlsplit(base)
        expected = urlunsplit(
            parts._replace(scheme=actual.scheme, netloc=actual.netloc)
        )
    following = next(
        (
            urljoin(base, h)
            for h in parser.hrefs
            if _identity(urljoin(base, h)) == _identity(expected)
        ),
        None,
    )
    if following is None:
        # Sites often omit default sort/filter values in an explicit next link.
        # Trust that control only within the expected listing route and host.
        target = urlsplit(expected)
        for href in parser.next_hrefs:
            candidate = urljoin(base, href)
            parts = urlsplit(candidate)
            if (parts.scheme, parts.netloc, parts.path.rstrip("/")) != (
                target.scheme,
                target.netloc,
                target.path.rstrip("/"),
            ):
                continue
            wanted = dict(parse_qsl(target.query))
            received = dict(parse_qsl(parts.query))
            if any(
                key in wanted and received.get(key) != wanted[key]
                for key in ("page", "p", "pg", "from")
            ):
                continue
            following = candidate
            break
    return PageInfo(terminal=following is None, next_url=following)


class PaginatedStream(ScrapeStream):
    def __init__(self, generator, summary):
        super().__init__(generator)
        self.summary = summary

    async def aclose(self):
        try:
            await super().aclose()
        finally:
            if self.summary.status in {"pending", "running"}:
                self.summary.status = "cancelled"


def paginated_stream(
    *,
    core,
    constructor,
    page_url,
    item_extractor,
    pages: PageCount = 1,
    start_page: int = 1,
    page_info=None,
    iterator_config=None,
    terminal_error=None,
    verify_first=False,
    deduplicate=False,
    item_identity=None,
):
    """Discover one source page at a time; Helper retains bounded item concurrency.

    The extractor runs once per response. Discovery captures that same response,
    so it never issues a second preflight request. No URL array is materialized.
    Page failures always remain visible, even if a caller selected SKIP: a missing
    page cannot be interpreted as successful exhaustion. Item loading follows the
    caller's configuration. Explicit source ordering remains available there.
    """
    validate_pagination(pages, start_page)
    config = iterator_config or make_iterator_config(
        page_error_mode=ErrorMode.RAISE, item_error_mode=ErrorMode.RAISE
    )

    # A skipped page would conceal a hole in a complete scan.
    def visible_mode(mode):
        value = ErrorMode(mode)
        return ErrorMode.YIELD if value is ErrorMode.SKIP else value

    config = replace(
        config,
        page_error_mode=visible_mode(config.page_error_mode),
        item_error_mode=visible_mode(config.item_error_mode),
    )
    summary = PaginationSummary()

    def visible_errors(handler):
        if handler is None:
            return None

        async def handle(context):
            action = handler(context)
            if isawaitable(action):
                action = await action
            # A custom SKIP cannot conceal a hole in a pagination scan.
            return ErrorAction.YIELD if action is ErrorAction.SKIP else action

        return handle

    config = replace(
        config,
        page_error_handler=visible_errors(config.page_error_handler),
        item_error_handler=visible_errors(config.item_error_handler),
    )

    async def generate():
        number = start_page
        current = page_url(number)
        visited, signatures, seen_items = set(), set(), set()
        summary.status = "running"
        can_end_on_error = verify_first and start_page > 1
        try:
            if verify_first and start_page > 1:
                first = await core.fetch_text(
                    page_url(1), method=config._page_request_method
                )
                _check_listing_response(first, page_url(1))
                first_rows = tuple(item_extractor(first))
                first_info = (
                    page_info(first, 1, first_rows)
                    if page_info
                    else PageInfo(terminal=not first_rows)
                )
                if (
                    first_info.total_pages is not None
                    and start_page > first_info.total_pages
                ) or (first_info.total_pages is None and first_info.terminal is True):
                    summary.status = "exhausted"
                    return
                first_signature = _page_signature(first_rows)
                if first_signature:
                    signatures.add(first_signature)
            while True:
                key = _identity(current)
                if key in visited:
                    raise ValueError("Pagination repeated a request")
                visited.add(key)
                captured = {}

                def extract(content):
                    _check_listing_response(content, current)
                    rows = tuple(item_extractor(content))
                    info = (
                        page_info(content, number, rows)
                        if page_info
                        else PageInfo(terminal=not rows)
                    )
                    if not isinstance(info, PageInfo):
                        raise TypeError("page_info must return PageInfo")
                    if info.terminal is not None and type(info.terminal) is not bool:
                        raise ValueError("Invalid terminal-page signal")
                    if type(info.next_is_probe) is not bool:
                        raise ValueError("Invalid next-page probe signal")
                    if info.next_url is not None and (
                        not isinstance(info.next_url, str) or not info.next_url.strip()
                    ):
                        raise ValueError("Invalid next-page URL")
                    if info.current_page is not None and (
                        type(info.current_page) is not int
                        or info.current_page != number
                    ):
                        raise ValueError("Server returned a different listing page")
                    if info.total_pages is not None:
                        if type(info.total_pages) is not int or info.total_pages < 0:
                            raise ValueError("Invalid total page count")
                        if number > info.total_pages:
                            if rows:
                                raise ValueError(
                                    "Server returned items beyond its total pages"
                                )
                            info = replace(info, terminal=True)
                        elif number == info.total_pages:
                            info = replace(info, terminal=True)
                        else:
                            # An empty/filtered source page can precede later
                            # items. A trusted indexed bound takes precedence.
                            info = replace(info, terminal=False)
                    signature = _page_signature(rows)
                    # A source can place the same cards on two distinct indexed
                    # pages. Require an independently parsed current-page signal
                    # before accepting that; URL generation alone proves nothing.
                    if (
                        signature
                        and signature in signatures
                        and info.current_page is None
                    ):
                        raise ValueError("Server repeated a complete listing page")
                    captured.update(info=info, signature=signature)
                    return rows

                stream = Helper(core=core, constructor=constructor).iterator(
                    [current], extract, iterator_config=config
                )
                try:
                    async with stream:
                        async for result in stream:
                            if result.error is not None:
                                if (
                                    can_end_on_error
                                    and terminal_error
                                    and terminal_error(result.error, number)
                                ):
                                    summary.status = (
                                        "exhausted"
                                        if not summary.errors
                                        else "incomplete"
                                    )
                                    return
                                summary.errors += 1
                            elif deduplicate:
                                identity = (
                                    item_identity(result.item)
                                    if item_identity
                                    else result.url
                                )
                                if identity in seen_items:
                                    summary.duplicates += 1
                                    continue
                                seen_items.add(identity)
                            yield replace(result, page_index=number - start_page)
                except Exception as error:
                    if (
                        can_end_on_error
                        and terminal_error
                        and terminal_error(error, number)
                    ):
                        summary.status = (
                            "exhausted" if not summary.errors else "incomplete"
                        )
                        return
                    raise
                if not captured:
                    summary.status = "incomplete"
                    return
                signatures.add(captured["signature"])
                summary.pages += 1
                summary.last_page = number
                info = captured["info"]
                if info.terminal is True:
                    summary.status = "exhausted" if not summary.errors else "incomplete"
                    return
                if pages != "all" and summary.pages >= pages:
                    summary.status = (
                        "limit_reached" if not summary.errors else "incomplete"
                    )
                    return
                if info.next_url is None and info.terminal is None:
                    raise ValueError(
                        "Provider cannot establish the next page or exhaustion"
                    )
                number += 1
                current = info.next_url or page_url(number)
                # A 404 after an advertised next link is a hole, not exhaustion.
                # Only an explicitly speculative boundary probe may end on it.
                can_end_on_error = info.next_is_probe
        except (GeneratorExit, asyncio.CancelledError):
            summary.status = "cancelled"
            raise
        except BaseException:
            summary.errors += 1
            summary.status = "failed"
            raise
        finally:
            if summary.status == "running":
                summary.status = "cancelled"

    return PaginatedStream(generate(), summary)


def http_status_is(error, status):
    """Inspect translated provider/Helper errors without losing HTTP semantics."""
    from base_api.modules.errors import HTTPStatusError

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, HTTPStatusError):
            return error.status_code == status
        error = error.__cause__ or error.__context__
    return False
