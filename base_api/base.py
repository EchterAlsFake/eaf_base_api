from __future__ import annotations

"""
The code in this library is insanely well documented. This was 50/50 by me and AI because this codebase got so complex
and handles so many edge cases I NEED this much detail in the docs to not get lost when comming back to this
after 2 weeks. 

Thanks for understanding :)
"""

import re
import os
import sys
import math
import json
import uuid
import time
import string
import shutil
import random
import asyncio
import logging
import traceback
import threading

from queue import Queue
from itertools import islice
from collections import deque
from functools import lru_cache
from urllib.parse import urljoin
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, cast, Union, Callable, Tuple, Iterable, TYPE_CHECKING, AsyncGenerator, Coroutine

from curl_cffi import CurlOpt # Used for DNS over HTTPS
from curl_cffi import requests
from curl_cffi.requests.errors import RequestsError
from curl_cffi.requests import AsyncSession, Response

if TYPE_CHECKING:
    from .modules.errors import *
    from .modules.type_hints import *
    from .modules.config import config, RuntimeConfig
    from .modules.progress_bars import Callback

else:
    try:
        from modules.errors import *
        from modules.type_hints import *
        from modules.config import config
        from modules.progress_bars import Callback
    except (ModuleNotFoundError, ImportError):
        from .modules.errors import *
        from .modules.type_hints import *
        from .modules.config import config
        from .modules.progress_bars import Callback

# The following imports are optional, because they depend on per API and I want to be as memory efficient as possible :)

try:
    import m3u8
    # Needed for all videos that use HLS streaming. Some do not and use mp4 containers / files instead
except (ModuleNotFoundError, ImportError):
    m3u8 = None  # type: ignore


UA_DESKTOP_CHROME = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
loggers: Dict[str, logging.Logger] = {}
HEIGHT_FROM_URI = re.compile(r'(?<!\d)(\d{3,4})[pP](?!\d)')  # e.g., 1080p, 720P
REGEX_CHALLENGE = re.compile(r'var p=(\d+); var s=(\d+);.*?(\d+):1;', re.DOTALL)


def eval_flags(flags: list[int]) -> int:
    """
    Evaluate flags.

    Args:
        flags (list[int]): List of flags arguments.

    Returns:
        int: The flag(s) value.
    """

    if len(flags):
        return flags[0]

    return 0


def subc(*args: Any) -> Callable[..., Any]:
    """
    Compile a substraction regex and apply its replacement to each call.

    Returns:
        Callable: Wrapped regex callable.
    """

    *flags, pattern, repl = args
    flags_val = eval_flags(flags)

    regex = re.compile(pattern, flags_val)

    def wrapper(*args: Any) -> Any:
        return regex.sub(repl, *args)

    return wrapper

parse_challenge = subc(re.DOTALL,              r'(?:var )|(?:/\*.*?\*/)|\s|\n|\t|(?:n;)', ''                                 ) # Parse challenge syntax
ponct_challenge = subc(re.DOTALL,              r'(if.*?&1\)|else)', r'\1:'                                                   ) # Convert challenge syntax



def least_factors(n: int) -> int:
    if n <= 0: return 0
    if n % 2 == 0: return 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return i
    return n



def _normalize_quality_value(quality: Union[str, int]) -> Union[str, int]:
    """
    quality: represents the quality value that should be normalized
    """
    if isinstance(quality, int):
        return quality # If the quality value is already an int, just return it directly

    quality = str(quality).lower().strip() # Convert to string, lower and remove white spaces

    if quality in {"best", "half", "worst"}:
        return quality # best, half and worst are also accepted values and will be further resolved in other functions

    m = re.search(r'(\d{3,4})', quality) # Search for int values that fit 144p-2160p values. if found: return as int
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid quality: {quality}")


def _choose_quality_from_list(available: List[str | int], target: Union[str, int]) -> int:
    # available like ["240", "360", "480", "720", "1080"] (Can also be unsorted)
    available_ints = sorted({int(x) for x in available}) # -> [144, 240, etc...]
    if isinstance(target, str):
        if target == "best":
            return available_ints[-1] # Return the last index, as this represents the best quality [144,360,720] -1 = 720
        if target == "worst":
            return available_ints[0] # Return the first index, as this represents the worst quality [144,360,720] 0 = 144
        if target == "half":
            return available_ints[len(available_ints) // 2] # Divides all options by 2 to find the middle, will round up though
        raise ValueError("Invalid label.")

    # numeric: highest ≤ target, else closest
    le = [h for h in available_ints if h <= target]

    """
    This works by iterating over the available qualities until the target is reached. It creates a new list with these
    qualities and returns the last index, as the last index in this case must be maximum best quality before we go over
    the specified one.
    """

    if le:
        return le[-1] # Returns the stuff
    # fallback closest (ties -> higher)
    # This happens if the user for example specified 144 as the quality, but only 240+ is available.
    return available_ints[0]


class ErrorVideo:
    """
    Basically, when I have all the async video objects flowing in and a video raises an error it would lead to massive
    problems for all future video objects.

    This way, we just return the Video back like normal, but give the error to all 'usual' attributes lol
    May not be the best solution, but works though
    """
    def __init__(self, url: str, err: Exception) -> None:
        self.url = url
        self._err = err

    def __getattr__(self, _: str) -> Any:
        # Any attribute access surfaces the original error
        raise self._err


def _height_from_variant(variant: Any) -> Optional[int]:
    """Extract height from a variant:
    1) stream_info.resolution (w, h)
    2) URI pattern like .../720p/...
    """
    if getattr(variant, "stream_info", None) and variant.stream_info.resolution:
        _, h = variant.stream_info.resolution  # (width, height)
        return int(h) # -> returns the height of a variant of an m3u8 master playlist

    # Fallback to search with a regex pattern
    if variant.uri:
        m = HEIGHT_FROM_URI.search(variant.uri)
        if m:
            return int(m.group(1))

    # If nothing is found, though this shouldn't happen
    return None

def _is_video_playlist(variant: Any) -> bool:
    """Filter out I-frames/audio-only playlists."""
    # m3u8 lib sometimes sets is_iframe if EXT-X-I-FRAME-STREAM-INF is present.
    if getattr(variant, "is_iframe", False):
        return False

    # If codecs known and contain only audio (mp4a, ac-3, ec-3, etc.)
    codecs = getattr(variant.stream_info, "codecs", None) if getattr(variant, "stream_info", None) else False
    if codecs:
        # very light heuristic: if no video codec substring, probably audio-only.
        # video: avc1, hvc1, hev1, vp9, av01, dvh
        assert isinstance(codecs, str)
        if not any(v in codecs.lower() for v in ("avc1", "hvc1", "hev1", "av01", "vp9", "dvh")):
            return False

    return True

def _collect_variants(master: Any) -> List[Dict[str, Any]]:
    """Normalize playlist variants to a comparable list."""
    items: List[Dict[str, Any]] = []
    for v in master.playlists:
        if not _is_video_playlist(v):
            continue

        h = _height_from_variant(v)
        bw = getattr(v.stream_info, "bandwidth", 0) if getattr(v, "stream_info", None) else 0
        fr = getattr(v.stream_info, "frame_rate", 0.0) if getattr(v, "stream_info", None) else 0.0
        items.append({
            "uri": v.uri,
            "height": h,                 # may be None
            "bandwidth": int(bw or 0),
            "frame_rate": float(fr or 0.0),
            "resolution": getattr(v.stream_info, "resolution", None) if getattr(v, "stream_info", None) else None,
            "raw": v
        })
    return items

def _pick_by_label(variants: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """best / worst / half based on a combined rank by (height, bandwidth)."""
    # rank by height first, then bandwidth as tiebreaker
    def key_fn(v: Dict[str, Any]) -> Tuple[int, int]:
        return v["height"] or 0, v["bandwidth"]
    ordered = sorted(variants, key=key_fn)

    if not ordered:
        raise ValueError("No video variants available in master playlist.")

    if label == "worst":
        return ordered[0]
    elif label == "half":
        return ordered[len(ordered)//2]
    elif label == "best":
        return ordered[-1]
    else:
        raise ValueError("Invalid quality label.")


def _pick_by_height(variants: List[Dict[str, Any]], target: int) -> Dict[str, Any]:
    """Choose the highest height ≤ target; else closest by absolute diff (ties -> higher)."""
    with_height = [v for v in variants if v["height"] is not None]
    if with_height:
        # Prefer height <= target
        below_eq = [v for v in with_height if v["height"] <= target]
        if below_eq:
            # Among same height, prefer higher bandwidth then higher fps
            best = sorted(below_eq, key=lambda v: (v["height"], v["bandwidth"], v["frame_rate"]))[-1]
            return best

        # Fallback: closest by absolute diff; ties -> higher height
        def diff_key(v: Dict[str, Any]) -> Tuple[int, int, int, float]:
            return abs((v["height"] or 0) - target), -(v["height"] or 0), v["bandwidth"], v["frame_rate"]
        return sorted(with_height, key=diff_key)[0]

    # If we have no heights at all, fall back to bandwidth ranking
    return sorted(variants, key=lambda v: v["bandwidth"])[-1]


# Default formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def is_android() -> bool:
    """Detects if the script is running on an Android device."""
    return "ANDROID_ROOT" in os.environ and "ANDROID_DATA" in os.environ

def get_log_file_path(filename: str = "app.log") -> str:
    """Returns a valid log file path that works on Android and other OS."""
    if is_android():
        return os.path.join(os.environ["HOME"], filename)  # Internal app storage
    return filename  # Default for Linux, Windows, Mac


def send_log_message(ip: str, port: Union[int, str], message: str) -> None:
    """Sends a log message to a remote server via HTTP or HTTPS."""
    try:
        url = f"https://{ip}:{port}/feedback"
        requests.post(url, json={"message": message}, timeout=5, impersonate="chrome110")
    except Exception as e:
        print(f"Failed to send log to {ip}:{port} - {e}", file=sys.stderr)


class HTTPLogHandler(logging.Handler):
    """Custom log handler that sends logs to a remote HTTP server."""
    def __init__(self, ip: str, port: Union[int, str]) -> None:
        super().__init__()
        self.ip = ip
        self.port = port

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        send_log_message(self.ip, self.port, log_entry)


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.CRITICAL, http_ip: Optional[str] = None, http_port: Optional[Union[int, str]] = None) -> logging.Logger:
    """Creates or updates a logger for a specific module."""
    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if name in loggers:
        logger = loggers[name]
        logger.setLevel(level)

        file_handler_exists = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        http_handler_exists = any(isinstance(h, HTTPLogHandler) for h in logger.handlers)

        if log_file and not file_handler_exists:
            log_file = get_log_file_path(log_file)
            fh = logging.FileHandler(log_file, mode='a')
            fh.setFormatter(logging.Formatter(format_))
            logger.addHandler(fh)

        if http_ip and http_port and not http_handler_exists:
            http_handler = HTTPLogHandler(http_ip, http_port)
            http_handler.setFormatter(logging.Formatter(format_))
            logger.addHandler(http_handler)

        return logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(format_))
        logger.addHandler(ch)

    if log_file:
        log_file = get_log_file_path(log_file)
        if not hasattr(sys, '_first_run'):
            setattr(sys, '_first_run', True)
            file_mode = 'w'
        else:
            file_mode = 'a'
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(logging.Formatter(format_))
        logger.addHandler(fh)

    if http_ip and http_port:
        http_handler = HTTPLogHandler(http_ip, http_port)
        http_handler.setFormatter(logging.Formatter(format_))
        logger.addHandler(http_handler)

    loggers[name] = logger
    return logger


class Cache:
    """
    Caches content from network requests
    """

    def __init__(self, configuration: "RuntimeConfig") -> None:
        self.cache_dictionary: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.logger = setup_logger("BASE API - [Cache]", level=logging.CRITICAL)
        self.configuration = configuration

    def enable_logging(self, log_file: Optional[str] = None, level: int = logging.DEBUG, log_ip: Optional[str] = None, log_port: Optional[Union[int, str]] = None) -> None:
        """
        Enables logging dynamically for this module.
        """
        self.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)

    def handle_cache(self, url: Optional[str]) -> Any:
        if url is None:
            return None

        with self.lock:
            content = self.cache_dictionary.get(url, None)
            return content

    def save_cache(self, url: str, content: Any) -> None:
        with self.lock:
            if len(self.cache_dictionary.keys()) >= self.configuration.max_cache_items:
                first_key = next(iter(self.cache_dictionary))
                # Delete the first item
                del self.cache_dictionary[first_key]
                self.logger.info(f"Deleting: {first_key} from cache, due to caching limits...")

            self.cache_dictionary[url] = content

    def save_segments_to_cache(self, m3u8_url: str, segments: List[Any]) -> None:
        with self.lock:
            self.cache_dictionary[m3u8_url] = segments

    def get_segments_from_cache(self, m3u8_url: str) -> Optional[List[Any]]:
        with self.lock:
            segments = self.cache_dictionary.get(m3u8_url, None)
            return segments

    def delete_cache(self, entry: str) -> None:
        with self.lock:
            self.cache_dictionary.pop(entry)

class Helper:
    def __init__(
        self,
        core: Any,
        video: Callable[..., Any],
        *,
        logger: Optional[logging.Logger] = None,
        log_name: str = "helper.iterator",
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        http_ip: Optional[str] = None,
        http_port: Optional[Union[int, str]] = None,
        other: Optional[Callable[..., Any]] = None
    ) -> None:
        """
        Args:
            core: object with .fetch(url) -> html
            video: callable Video(url, core=...)
            logger: optional pre-configured logger (if not provided, one is created via setup_logger)
            log_name/log_file/log_level/http_*: used only when `logger` is None
        """
        super().__init__()
        self.core = core
        self.Video = video
        self.OtherReturn = other

        if logger is not None:
            self.logger = logger
        else:
            # Create a dedicated, re-usable logger; safe to call multiple times thanks to setup_logger
            self.logger = setup_logger(
                name=log_name,
                log_file=log_file,
                level=log_level,
                http_ip=http_ip,
                http_port=http_port,
            )

    @staticmethod
    def chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
        """
        This function is used to limit page fetching, so that not all pages are fetched at once.
        Now with trace logs about chunk formation and exhaustion.
        """
        if size <= 0:
            raise ValueError("chunk size must be > 0")
        it = iter(iterable)
        idx = 0
        logger = logging.getLogger("helper.iterator")
        is_debug = logger.isEnabledFor(logging.DEBUG)
        while True:
            block = list(islice(it, size))
            if not block:
                return
            # Logging kept staticmethod-safe by using root logger to avoid needing self
            if is_debug:
                logger.debug(
                    "chunked: yielding block #%d with %d items", idx, len(block)
                )
            idx += 1
            yield block

    def _get_video(self, url: str) -> Any:
        return self.Video(url, core=self.core)

    async def _other_return(self, url: str) -> Any:
        """In rare cases e.g., Xhamster we don't always return video objects with the iterator"""
        assert self.OtherReturn is not None
        v = self.OtherReturn(url, core=self.core)
        if asyncio.iscoroutine(v):
            v = await v
        return v

    async def _make_video_safe(self, url: str) -> Any:
        # Small helper wrapped with verbose logging
        logger = self.logger
        start = time.perf_counter()
        try:
            html = await self.core.fetch(url)
            v = self.Video(url, core=self.core, html_content=html)
            if asyncio.iscoroutine(v):
                v = await v
            dur = (time.perf_counter() - start) * 1000
            logger.debug("video_init ok url=%s (%.2f ms)", url, dur)
            return v
        except Exception as e:
            dur = (time.perf_counter() - start) * 1000
            logger.exception("video_init FAILED url=%s (%.2f ms): %s", url, dur, e)
            return ErrorVideo(url, e)

    async def iterator(
            self,
            page_urls: Optional[List[str]] = None,
            extractor: Optional[Callable[..., Any]] = None,
            pages_concurrency: int = 5,
            videos_concurrency: int = 20,
            other_return: bool = False,
            method_pages: str = "GET", # Only for very special use cases like xvideos account history
            method_videos: str = "GET",
    ) -> AsyncGenerator[Any, None]:
        """
        Yields Video/ErrorVideo in deterministic (page_idx, vid_idx) order while fetching concurrently.
        Stops scraping pages once a 404 page is encountered (self.core.fetch returns a Response).
        """
        logger = self.logger
        run_id = uuid.uuid4().hex[:8]  # correlate all logs for this iterator call
        t0 = time.perf_counter()

        if page_urls is None:
            raise ValueError("page_urls must be provided")
        if extractor is None:
            raise ValueError("extractor must be provided")
        if videos_concurrency < 1:
            raise ValueError("videos_concurrency must be >= 1")  # important to avoid deadlock

        # Cast to satisfy type checker
        other_return = bool(other_return)

        logger.info(
            "[%s] iterator start pages=%d pages_conc=%d videos_conc=%d",
            run_id, len(page_urls), pages_concurrency, videos_concurrency
        )

        # Results: (page_idx, vid_idx) -> Video/ErrorVideo
        results: Dict[Tuple[int, int], Any] = {}
        # Count of videos per page
        page_counts: Dict[int, int] = {}

        next_page_idx = 0  # ordering cursor
        next_video_idx = 0

        # NEW: stop paging after first 404
        stop_after_404 = False
        stop_at_page_idx: Optional[int] = None

        def flush_ready() -> List[Any]:
            nonlocal next_page_idx, next_video_idx
            items_to_yield: List[Any] = []
            flushed = 0
            while True:
                if next_page_idx not in page_counts:
                    if flushed:
                        logger.debug(
                            "[%s] flush_ready paused: next_page=%d unknown count; flushed=%d",
                            run_id, next_page_idx, flushed
                        )
                    return items_to_yield
                if next_video_idx >= page_counts[next_page_idx]:
                    logger.debug(
                        "[%s] page complete pidx=%d total_videos=%d -> advancing",
                        run_id, next_page_idx, page_counts[next_page_idx]
                    )
                    next_page_idx += 1
                    next_video_idx = 0
                    continue

                key = (next_page_idx, next_video_idx)
                if key not in results:
                    if flushed:
                        logger.debug(
                            "[%s] flush_ready stopping: awaiting key=%s; flushed=%d",
                            run_id, key, flushed
                        )
                    return items_to_yield

                x_item = results.pop(key)
                logger.debug(
                    "[%s] flush_ready yield pidx=%d vidx=%d (remaining_results=%d)",
                    run_id, key[0], key[1], len(results)
                )
                items_to_yield.append(x_item)
                flushed += 1
                next_video_idx += 1

            return []

        page_iter = iter(enumerate(page_urls))

        page_in_flight: Dict[asyncio.Task, Tuple[int, str]] = {}
        video_in_flight: Dict[asyncio.Task, Tuple[int, int]] = {}
        pending_videos: deque[Tuple[int, int, str]] = deque()

        def schedule_videos() -> int:
            scheduled = 0
            while pending_videos and len(video_in_flight) < videos_concurrency:
                pidx, vid_idx, vurl = pending_videos.popleft()
                if other_return:
                    task = asyncio.create_task(self._other_return(vurl))
                else:
                    task = asyncio.create_task(self._make_video_safe(vurl))
                video_in_flight[task] = (pidx, vid_idx)
                scheduled += 1
                logger.debug(
                    "[%s] scheduled VIDEO pidx=%d vidx=%d vurl=%s "
                    "(inflight_video=%d queued=%d)",
                    run_id, pidx, vid_idx, vurl, len(video_in_flight), len(pending_videos)
                )
            return scheduled

        # Prime initial page fetches
        for _ in range(pages_concurrency):
            try:
                pidx, url = next(page_iter)
            except StopIteration:
                break
            task = asyncio.create_task(self.core.fetch(url, method=method_pages))
            page_in_flight[task] = (pidx, url)
            logger.debug(
                "[%s] scheduled PAGE pidx=%d url=%s (queue=%d)",
                run_id, pidx, url, len(page_in_flight)
            )

        # Main loop
        while page_in_flight or video_in_flight or pending_videos:
            # Fill any free video slots before waiting
            if pending_videos and len(video_in_flight) < videos_concurrency:
                schedule_videos()

            waiting_on = [*page_in_flight, *video_in_flight]
            logger.debug(
                "[%s] waiting futures page=%d video=%d queued_videos=%d",
                run_id, len(page_in_flight), len(video_in_flight), len(pending_videos)
            )

            # If nothing in flight but items queued (shouldn't happen often), try to schedule then continue
            if not waiting_on:
                schedule_videos()
                if not (page_in_flight or video_in_flight):
                    break
                waiting_on = [*page_in_flight, *video_in_flight]

            done, _ = await asyncio.wait(waiting_on, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                # PAGE completed
                if task in page_in_flight:
                    pidx, url = page_in_flight.pop(task)
                    try:
                        html_or_resp = task.result()
                        logger.info(
                            "[%s] PAGE done pidx=%d url=%s (inflight_page=%d inflight_video=%d)",
                            run_id, pidx, url, len(page_in_flight), len(video_in_flight)
                        )
                    except Exception as e:
                        logger.exception(
                            "[%s] PAGE FAILED pidx=%d url=%s: %s",
                            run_id, pidx, url, e
                        )
                        # mark page as having 0 videos so flush/ordering moves on
                        page_counts[pidx] = 0
                        # now-ordered items may be ready
                        for item in flush_ready():
                            yield item
                        # schedule next page if available (unless we've already stopped)
                        if not stop_after_404:
                            try:
                                npidx, nurl = next(page_iter)
                                nfut = asyncio.create_task(self.core.fetch(nurl, method=method_pages))
                                page_in_flight[nfut] = (npidx, nurl)
                                logger.debug(
                                    "[%s] scheduled NEXT PAGE pidx=%d url=%s after failure",
                                    run_id, npidx, nurl
                                )
                            except StopIteration:
                                pass
                        continue

                    # Detect a 404 "page" (Response returned)
                    is_404 = False
                    try:
                        if isinstance(html_or_resp, Response) and html_or_resp.status_code == 404:
                            is_404 = True
                    except Exception:
                        # Fallback duck-typing if import fails for some reason
                        if getattr(html_or_resp, "status_code", None) == 404 and hasattr(html_or_resp, "headers"):
                            is_404 = True

                    if is_404:
                        logger.warning(
                            "[%s] PAGE 404 pidx=%d url=%s -> stopping further page scheduling",
                            run_id, pidx, url
                        )
                        stop_after_404 = True
                        stop_at_page_idx = pidx
                        # 404 page has 0 videos
                        page_counts[pidx] = 0
                        # Flush anything now ready
                        for item in flush_ready():
                            yield item
                        # Do NOT schedule any more pages
                        continue

                    # If we've already hit a 404 and this page index is after it, ignore this page
                    if stop_after_404 and stop_at_page_idx is not None and pidx > stop_at_page_idx:
                        logger.info(
                            "[%s] ignoring PAGE pidx=%d url=%s (after first 404 at pidx=%d)",
                            run_id, pidx, url, stop_at_page_idx
                        )
                        page_counts[pidx] = 0
                        for item in flush_ready():
                            yield item
                        continue

                    # Normal path: Extract video URLs
                    html = html_or_resp
                    try:
                        res = extractor(html)
                        if asyncio.iscoroutine(res):
                            video_urls = await res
                        else:
                            video_urls = res
                        video_urls = video_urls or []
                        logger.debug(
                            "[%s] extractor ok pidx=%d urls=%d",
                            run_id, pidx, len(video_urls)
                        )
                    except Exception as e:
                        logger.exception(
                            "[%s] extractor FAILED pidx=%d url=%s: %s",
                            run_id, pidx, url, e
                        )
                        video_urls = []

                    page_counts[pidx] = len(video_urls)
                    logger.info(
                        "[%s] PAGE indexed pidx=%d video_count=%d",
                        run_id, pidx, page_counts[pidx]
                    )

                    for vid_idx, vurl in enumerate(video_urls):
                        pending_videos.append((pidx, vid_idx, vurl))
                    schedule_videos()

                    for item in flush_ready():
                        yield item

                    # Fetch next page only if we haven't seen a 404
                    if not stop_after_404:
                        try:
                            npidx, nurl = next(page_iter)
                            ntask = asyncio.create_task(self.core.fetch(nurl, method=method_videos))
                            page_in_flight[ntask] = (npidx, nurl)
                            logger.debug(
                                "[%s] scheduled NEXT PAGE pidx=%d url=%s (inflight_page=%d)",
                                run_id, npidx, nurl, len(page_in_flight)
                            )
                        except StopIteration:
                            logger.debug("[%s] no more pages to schedule", run_id)

                # VIDEO completed
                elif task in video_in_flight:
                    pidx, vid_idx = video_in_flight.pop(task)
                    try:
                        val = task.result()
                        results[(pidx, vid_idx)] = val
                        logger.debug(
                            "[%s] VIDEO done pidx=%d vidx=%d (inflight_video=%d results_cached=%d)",
                            run_id, pidx, vid_idx, len(video_in_flight), len(results)
                        )
                    except Exception as e:
                        logger.exception(
                            "[%s] VIDEO FAILED pidx=%d vidx=%d: %s",
                            run_id, pidx, vid_idx, e
                        )
                        results[(pidx, vid_idx)] = ErrorVideo(f"<unknown:{pidx}/{vid_idx}>", e)

                    # Flush ordered results, then backfill video slots
                    for item in flush_ready():
                        yield item
                    schedule_videos()

        # Final flush (should be no-ops, but logged for completeness)
        logger.debug("[%s] final flush start", run_id)
        for item in flush_ready():
            yield item
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("[%s] iterator complete in %.2f ms", run_id, total_ms)


def async_generator_to_sync(async_gen_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
    """
    Safely bridges an async generator to a synchronous generator using
    a background thread and a Queue.
    """
    sync_queue: Queue[Any] = Queue()
    sentinel = object()  # Used to signal the generator is finished
    error_sentinel = object()  # Used to signal an exception occurred

    def _background_runner() -> None:
        # Create a new event loop for this background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _consume() -> None:
            try:
                # Instantiate and iterate over the async generator
                gen = async_gen_func(*args, **kwargs)
                async for item in gen:
                    sync_queue.put(item)
            except Exception as e:
                # Catch any errors in the scraper and pass them to the sync thread
                sync_queue.put((error_sentinel, e))
            finally:
                # Signal that we are done
                sync_queue.put(sentinel)

        try:
            loop.run_until_complete(_consume())
        finally:
            loop.close()

    # 1. Start the async execution in a background thread
    thread = threading.Thread(target=_background_runner, daemon=True)
    thread.start()

    # 2. Synchronously yield from the queue as items come in
    while True:
        item = sync_queue.get()  # This blocks until an item is ready!

        # Check if the async generator finished
        if item is sentinel:
            break

        # Check if the async generator threw an exception
        if isinstance(item, tuple) and len(item) == 2 and item[0] is error_sentinel:
            raise item[1]

        yield item

    thread.join()


class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(self, config: "RuntimeConfig" = config) -> None:
        self.lock = asyncio.Lock()
        self.latest_key: Optional[str] = None
        self.latest_key_time: float = 0.0
        self.last_request_time = time.time()
        self.total_requests: int = 0  # Tracks how many requests have been made
        self.session: Optional[AsyncSession] = None
        self.config = config
        self.cache = Cache(self.config)
        self.logger = setup_logger("BASE API - [BaseCore]", log_file=None, level=logging.ERROR)
        self.default_headers = {
            "User-Agent": UA_DESKTOP_CHROME,
            "Accept-Language": self.config.locale,
            "Accept-Encoding": "gzip, deflate, br"
        }

    def enable_logging(self, log_file: Optional[str] = None, level: int = logging.DEBUG, log_ip: Optional[str] = None, log_port: Optional[Union[int, str]] = None) -> None:
        """Enables logging dynamically for this module."""
        self.logger = setup_logger(name="BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)
        self.cache.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)

    def initialize_session(self) -> None:
        verify = self.config.verify_ssl

        curl_options: Dict[CurlOpt, Union[bytes, int]] = {}
        if self.config.dns_over_https:
            curl_options[CurlOpt.DOH_URL] = str(self.config.dns_over_https).encode("utf-8")

        proxies = None
        if self.config.proxies:
            proxies = self.config.proxies

        if self.config.max_bandwidth_mb is not None and self.config.max_bandwidth_mb > 0:
            global_limit_bytes = int(self.config.max_bandwidth_mb * 1024 * 1024)
            total_concurrent_connections = self.config.max_workers_download * self.config.videos_concurrency
            per_connection_limit = max(1, int(global_limit_bytes / total_concurrent_connections))
            curl_options[CurlOpt.MAX_RECV_SPEED_LARGE] = per_connection_limit

        js3 = self.config.custom_ja3
        impersonation = self.config.impersonation
        http_version = self.config.http_version
        proxy_auth_str = self.config.proxy_auth
        trust_env = self.config.trust_env

        p_auth: Optional[Tuple[str, str]] = None
        if proxy_auth_str and ":" in proxy_auth_str:
            u, p = proxy_auth_str.split(":", 1)
            p_auth = (u, p)

        self.session = cast(Any, AsyncSession)(
            proxies=cast(Any, proxies),
            timeout=self.config.timeout,
            verify=verify,
            impersonate=cast(Any, impersonation),
            curl_options=curl_options,
            http_version=cast(Any, http_version),
            ja3=js3,
            proxy_auth=p_auth,
            trust_env=trust_env
        )
        # Ensure our defaults are on the session
        self.session.headers.update(self.default_headers)

    async def enforce_delay(self) -> None:
        """Enforces the specified delay in config.request_delay (only if > 0)."""
        delay = self.config.request_delay
        if delay and delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            self.logger.debug(f"Time since last request: {time_since_last_request:.2f} seconds.")
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                self.logger.debug(f"Enforcing delay of {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()

    def _merged_headers(self, override: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """
        Create request headers from current session headers + optional overrides.
        Overrides win, session headers are the base.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        headers: Dict[str, Any] = cast(Dict[str, Any], cast(Any, dict(session.headers)))
        if override:
            headers.update(override)
        return headers

    def _merged_cookies(self, override: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Same as above, but for cookies"""
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        cookies: Dict[str, Any] = cast(Dict[str, Any], cast(Any, dict(session.cookies)))
        if override:
            cookies.update(override)
        return cookies

    def _parse_retry_after(self, response: Response) -> Optional[float]:
        """Parse Retry-After (seconds or http-date) into seconds; None if not present/invalid."""
        v = response.headers.get("Retry-After")
        if not v:
            return None
        try:
            # numeric seconds
            return float(v)
        except ValueError:
            try:
                dt = parsedate_to_datetime(v)
                # Convert to seconds from now
                delta = (dt - dt.now(dt.tzinfo)).total_seconds()
                # clamp: negative -> 0
                return max(0.0, delta)
            except Exception:
                return None

    def _format_headers_for_log(self, headers: Any) -> Dict[str, str]:
        """Redact sensitive headers but keep enough signal for debugging."""
        sensitive = {
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "x-auth-token",
            "x-csrf-token",
            "x-xsrf-token",
        }
        out: Dict[str, str] = {}
        for key, value in headers.items():
            lkey = key.lower()
            if lkey in sensitive:
                if lkey == "cookie":
                    parts = [p.split("=", 1)[0].strip() for p in str(value).split(";") if p.strip()]
                    value = f"<redacted:{','.join(parts)}>" if parts else "<redacted>"
                else:
                    value = "<redacted>"
            if key in out:
                out[key] = f"{out[key]}, {value}"
            else:
                out[key] = str(value)
        return out

    def _response_body_preview(self, response: Response, max_bytes: int = 512) -> str:
        try:
            raw = response.content[:max_bytes]
        except Exception as e:
            return f"<failed to read body: {e}>"
        if not raw:
            return "<empty>"
        enc = getattr(response, "encoding", None) or "utf-8"
        try:
            text = cast(bytes, cast(Any, raw)).decode(enc, errors="replace")
        except Exception:
            text = cast(bytes, cast(Any, raw)).decode("utf-8", errors="replace")
        return text.replace("\r", "\\r").replace("\n", "\\n")

    def _log_precondition_failed(self, response: Response, attempt: int) -> None:
        req = response.request
        try:
            req_headers = self._format_headers_for_log(req.headers) if req is not None else {}
        except Exception as e:
            req_headers = {"<error>": f"failed to format request headers: {e}"}

        try:
            resp_headers = self._format_headers_for_log(response.headers)
        except Exception as e:
            resp_headers = {"<error>": f"failed to format response headers: {e}"}

        try:
            cond_headers = [
                k for k in req.headers.keys() if k.lower().startswith("if-")
            ] if req is not None else []
        except Exception:
            cond_headers = []

        cond_note = f" conditional_headers={cond_headers}" if cond_headers else ""
        body_preview = self._response_body_preview(response)

        self.logger.warning(
            "HTTP 412 precondition failed (attempt %d) for %s %s.%s request_headers=%s response_headers=%s body_preview=%s",
            attempt + 1,
            getattr(req, "method", "UNKNOWN") if req is not None else "UNKNOWN",
            response.url,
            cond_note,
            req_headers,
            resp_headers,
            body_preview,
        )

    async def fetch(
        self,
        url: str,
        get_bytes: bool = False,
        timeout: Optional[int] = None,
        get_response: bool = False,
        save_cache: bool = True,
        cookies: Optional[Dict[str, str]] = None,
        allow_redirects: bool = True,
        data: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Union[bytes, str, Response]:
        """
        Fetch content with retries, optional caching, proxy support, and bandwidth limiting.

        Returns:
            - Response if get_response=True
            - bytes            if get_bytes=True
            - str (text)       otherwise

        Raises:
            - ResourceGone, ProxySSLError, KillSwitch, UnknownError
            - RequestsError on unrecoverable HTTP status (e.g., 4xx/5xx after retries, 403/429 exhausted)
            - RequestsError / Timeout errors may bubble up if not recoverable
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        # Cache (only for text mode)
        cache_hit = self.cache.handle_cache(url)
        if cache_hit is not None and not get_bytes and not get_response:
            self.logger.info(f"Fetched content for: {url} from cache!")
            return cache_hit

        req_timeout = timeout or self.config.timeout
        last_response: Optional[Response] = None

        max_retries = max(1, int(self.config.max_retries))
        for attempt in range(max_retries):
            # backoff (attempt 0 has no extra sleep)
            if attempt >= 1:
                # capped exponential backoff with jitter
                base = min(5.0, 0.5 * (2 ** attempt))
                jitter = random.random() * 0.25  # 0-250ms
                await asyncio.sleep(base + jitter)

            try:
                # Only applies if you explicitly set a delay in config (as you noted)
                await self.enforce_delay()

                # Recompute headers each attempt so changes (like UA switch) take effect
                req_headers = self._merged_headers(headers)
                req_cookies = self._merged_cookies(cookies)
                self.logger.debug(f"Using Headers: {req_headers}")
                self.logger.debug(f"Using Cookies: {req_cookies}")

                if isinstance(self.config.max_bandwidth_mb, int):
                    speed_limit = self.config.max_bandwidth_mb * 1024 * 1024 # Convert to bytes

                else:
                    speed_limit = None

                current_time = asyncio.get_event_loop().time()
                latest_key = self.latest_key
                if "KEY" not in session.cookies and latest_key is not None:
                    if current_time - self.latest_key_time < 10:  # 10-second freshness window
                        session.cookies.set("KEY", latest_key, domain=".pornhub.com", path="/")

                response = await cast(Any, session).request(
                    method=cast(Any, method),
                    url=url,
                    timeout=req_timeout,
                    allow_redirects=allow_redirects,
                    data=data,
                    json=json,
                    params=params,
                    headers=req_headers,
                    cookies=req_cookies,
                    max_recv_speed=speed_limit or 0,
                )

                last_response = response
                self.total_requests += 1
                status = response.status_code

                content_type = response.headers.get("content-type", "").lower()
                is_html = "text/html" in content_type if content_type else (not get_bytes)

                if is_html:
                    enc = getattr(response, "encoding", None) or "utf-8"
                    resp_text = cast(bytes, cast(Any, response.content)).decode(enc, errors="replace")

                    if 'onload="go()"' in resp_text:
                        # Snapshot our persistent tracker instead of the volatile cookie jar
                        local_latest = getattr(self, "latest_key", None)

                        async with self.lock:
                            # DOUBLE CHECK: If the persistent key changed while we waited,
                            # another task successfully solved a newer challenge phase!
                            if getattr(self, "latest_key", None) != local_latest:
                                self.logger.info(
                                    "Another task already resolved the challenge! Retrying request with the new cookie.")
                                # Force ensure it's in the jar before looping back
                                if self.latest_key:
                                    session.cookies.set("KEY", self.latest_key, domain=".pornhub.com", path="/")
                                continue

                            self.logger.info("Challenge page detected! Solving...")

                            get_challenge = re.compile(r'go\(\).*?{(.*?)n=l.*?KEY.*?s\+\":(\d+):', re.DOTALL)
                            challenge_data = re.search(get_challenge, resp_text)

                            if challenge_data:
                                try:
                                    challenge_str, token_str = challenge_data.groups()

                                    code = parse_challenge(challenge_str)
                                    code = ponct_challenge(code)
                                    code = '\n'.join(code.split(';'))

                                    # Sanitizing inputs to prevent possible RCE
                                    safe_chars = set(string.ascii_letters + string.digits + " \t\n=+-*/().:><&|~^")
                                    if not all(c in safe_chars for c in code):
                                        self.logger.error(f"Security Abort: Challenge contains illegal characters. CODE: {code}")
                                        raise SecurityAbort

                                    # Additional Sandboxing for exec environment
                                    safe_globals: Dict[str, Any] = {"__builtins__": {}}
                                    safe_locals = {"p": 0, "s": 0}

                                    # Execute the code in the completely isolated sandbox
                                    exec(code, safe_globals, safe_locals)

                                    # Safely retrieve the variables using .get() to prevent KeyErrors if the challenge format changes
                                    p = safe_locals.get('p', 0)
                                    s = safe_locals.get('s', 0)
                                    n = least_factors(p)

                                    cookie_value = f'{n}*{p // n}:{s}:{token_str}:1'

                                    # Update BOTH our internal tracker and the global session jar
                                    self.latest_key = cookie_value
                                    self.latest_key_time = asyncio.get_event_loop().time()

                                    session.cookies.set("KEY", cookie_value, domain=".pornhub.com", path="/")
                                    self.logger.info(f"RESOLVED CHALLENGE! Injected cookie: {cookie_value}")

                                    try:
                                        self.cache.delete_cache(url)
                                    except (KeyError, Exception):
                                        pass

                                    await asyncio.sleep(1.5)
                                    continue

                                except Exception as math_err:
                                    self.logger.error(f"Failed executing math engine: {repr(math_err)}")
                                    continue
                            else:
                                self.logger.error("Detected challenge page, but your regex failed to extract data.")
                                await asyncio.sleep(2)
                                continue
                # Fast path
                if status == 200:
                    self.logger.debug(f"Attempt {attempt}: Successfully fetched URL: {url}")

                    if get_response:
                        return response

                    raw_content = response.content

                    content: Union[str, bytes]
                    if get_bytes:
                        content = raw_content

                    else:
                        # Prefer server-provided/guessed encoding; fallback to utf-8 then latin-1
                        enc = getattr(response, "encoding", None) or "utf-8"
                        try:
                            content = cast(bytes, cast(Any, raw_content)).decode(enc, errors="strict")
                        except UnicodeDecodeError:
                            self.logger.warning(f"Content could not be decoded as {enc} ({url}), decoding in 'latin1' instead!")
                            content = cast(bytes, cast(Any, raw_content)).decode("latin1", errors="replace")
                        if save_cache:
                            self.logger.debug(f"Saving content of {url} to local cache.")
                            self.cache.save_cache(url, content)

                    return content

                elif status == 204:
                    return response

                # 403 handling: try a UA switch once, then give up
                if status == 403:
                    if attempt == 0 or attempt == 1:
                        # sleep briefly then switch UA for the next try
                        await asyncio.sleep(2)
                        session.headers.update({
                            "User-Agent": "AppleWebKit/537.36 (KHTML, like Gecko)"
                        })
                        self.logger.warning(f"Switched User-Agent to: {session.headers.get('User-Agent')}")
                        # continue to retry with new UA
                        continue

                    else:
                        # After at least one retry with new UA, fail
                        msg = f"Forbidden (403) after {attempt+1} attempts for URL: {url}"
                        self.logger.error(msg)
                        response.raise_for_status()

                if status == 412:
                    self._log_precondition_failed(response, attempt)

                # 404: usually not recoverable, but we try once (if attempt==0) in case of transient CDN
                if status == 404:
                    return response # Immediately returning response, because 404 usually means no more content when searching

                # 410: permanent gone
                if status == 410:
                    raise ResourceGone(f"Resource gone (HTTP 410) for URL: {url}") # Not gonna fix that, bro just reinitialize video and that's it lmao

                # 429: rate limited — respect Retry-After if present, else backoff and retry up to cap
                if status == 429:
                    wait = self._parse_retry_after(response)
                    if wait is None:
                        # fall back to exponential backoff proportional to attempt
                        wait = min(30.0, 0.5 * (2 ** attempt)) + random.random() * 0.5
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Rate limited (429). Waiting {wait:.2f}s then retrying ({attempt+1}/{max_retries}) for {url}.")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        self.logger.error(f"Rate limited (429) after {max_retries} attempts for {url}.")
                        return response

                # 5xx: transient server errors — retry until we run out
                if 500 <= status < 600:
                    self.logger.warning(f"Server error {status} on {url}. Retrying ({attempt+1}/{max_retries})...")
                    time.sleep(1)
                    continue

                # Other non-200s: let curl_cffi raise a typed error
                if status != 200:
                    self.logger.info(f"HTTP {status} for {url} (attempt {attempt+1}/{max_retries}).")
                    if attempt < max_retries - 1:
                        continue
                    return response

            except RequestsError as e:
                err_str = str(e).lower()
                self.logger.error(f"Attempt {attempt}: Request error for URL {url}: {e}")
                
                if "certificate verify failed" in err_str:
                    raise ProxySSLError("Proxy has an invalid SSL certificate, set 'verify = False' in config")
                elif "cookie conflict" in err_str:
                    self.logger.error(f"Cookie conflict. Aborting this request. Details: {e}")
                    raise UnknownError(f"Cookie conflict during request to {url}: {e}") from e
                elif "proxy" in err_str:
                    self.logger.error(f"Proxy Error for {url}: {e}")
                    raise KillSwitch("Proxy error when trying a request, aborting!") from e
                elif "timeout" in err_str or "read" in err_str:
                    self.logger.error(
                        f"Attempt {attempt}: Timeout for URL {url}: {e}. "
                        f"Consider increasing the timeout or check your connection."
                    )
                
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying ({attempt+1}/{max_retries}) for URL: {url}")
                    continue
                raise

            except ResourceGone:
                # propagate as-is
                raise

            except Exception as e:
                # Preserve original exception context
                self.logger.error(
                    f"Attempt {attempt}: Unexpected error for {url}: {e}\n{traceback.format_exc()}"
                )
                raise UnknownError(f"Unexpected error for URL {url}: {e}") from e

        # If we get here, we exhausted retries without returning or raising a status error.
        self.logger.error(f"Failed to fetch URL {url} after {max_retries} attempts.")
        if last_response is not None:
            # Raise a typed error with the last response if we have one.
            try:
                last_response.raise_for_status()
            except Exception as e:
                raise e
        # Otherwise raise a generic failure
        raise UnknownError(
            f"Failed to fetch: {url} after {max_retries} attempts. "
            "If you're sure you're not blocked and your connection is stable, "
            "please open an issue with the URL and steps to reproduce."
        )

    @classmethod
    def strip_title(cls, title: str, max_length: int = 255) -> str:
        """
        Sanitize a filename to be safe across Windows, macOS, Linux, and Android.
        Replaces or strips illegal characters and trims to a safe length.
        """

        # Reserved characters on Windows + `/` for Unix/macOS
        illegal_chars = r'[<>:"/\\|?*\x00-\x1F]'
        sanitized = re.sub(illegal_chars, "_", title)

        # Strip invisible zero-width characters
        sanitized = re.sub(r'[\u200B-\u200D\uFEFF]', '', sanitized)

        # Strip trailing periods or spaces (Windows)
        sanitized = sanitized.rstrip(" .")

        # Prevent reserved Windows filenames
        reserved_names = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        name_only = sanitized.split('.')[0].upper()
        if name_only in reserved_names:
            sanitized = f"_{sanitized}"

        # Trim to max length
        return sanitized[:max_length]

    @lru_cache(maxsize=250)
    async def get_m3u8_by_quality(self, m3u8_url: str, quality: Union[str, int]) -> str:
        """
        Return the media-playlist URL for the requested quality.

        quality:
          - 'best' | 'half' | 'worst'
          - 1080 / '1080' / '1080p' (and similar)
        """
        if m3u8 is None:
            raise ModuleNotFoundError(f"""
Using m3u8 is optional depending whether you use HLS videos or static videos. It seems like you are trying to download
from HLS. Please install m3u8 using: `pip install m3u8`.

If this does not fix the issue, there's an import error related to your environment. In this case please create
a new Python file, import only m3u8 and see what error you get. 
""")

        # Resolve master content
        assert m3u8 is not None
        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)
            self.logger.debug("Resolved inline/custom m3u8 master content.")
            base_for_join = ""  # URIs should be absolute in inline cases; join will handle if relative
        else:
            content = await self.fetch(url=m3u8_url)
            assert isinstance(content, str)
            master = m3u8.loads(content)
            base_for_join = m3u8_url
            self.logger.debug(f"Resolved m3u8 master: {m3u8_url}")

        if not master.is_variant:
            raise ValueError("Provided URL/content is not a master playlist.")

        variants = _collect_variants(master)
        if not variants:
            raise ValueError("No usable video variants found in master playlist.")

        q = _normalize_quality_value(quality)
        if isinstance(q, str):  # 'best'/'half'/'worst'
            chosen = _pick_by_label(variants, q)
        else:  # numeric height like 1080, 720, etc.
            chosen = _pick_by_height(variants, q)

        full_url = urljoin(base_for_join or m3u8_url, chosen["uri"])
        return full_url

    async def list_available_qualities(self, m3u8_url: str) -> List[int]:
        """
        Inspect the master playlist and return sorted unique heights (e.g., [240, 360, 480, 720, 1080]).
        """
        assert m3u8 is not None
        if not m3u8_url.startswith("https://"):
            master = m3u8.loads(m3u8_url)
        else:
            content = await self.fetch(url=m3u8_url)
            assert isinstance(content, str)
            master = m3u8.loads(content)

        if not master.is_variant:
            return []

        heights = {h for h in (_height_from_variant(v) for v in master.playlists) if h is not None}
        if heights:
            return sorted(heights)
        # fallback: bandwidth-only (roughly infer tiers)
        by_bw = sorted(
            (getattr(v.stream_info, "bandwidth", 0) for v in master.playlists if _is_video_playlist(v)),
            key=int
        )
        # Return rank numbers instead of heights if we truly can't infer—kept simple:
        return [i for i, _ in enumerate(by_bw, start=1)]

    async def get_segments(self, m3u8_url_master: str, quality: Union[str, int]) -> List[str]:
        assert m3u8 is not None
        _cache_url = f"{m3u8_url_master}{quality}"
        _segments: Optional[List[str]] = cast(Optional[List[str]], self.cache.get_segments_from_cache(_cache_url))
        if _segments is not None:
            self.logger.info(f"Received: {len(_segments)} from cache!")
            return _segments

        # Resolve the quality-specific playlist URL (may still be a master in some edge cases)
        playlist_url = await self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        self.logger.debug(f"Trying to fetch segments from m3u8 -> {playlist_url}")

        # M3U8s are volatile → don't cache
        content = await self.fetch(url=playlist_url, save_cache=False)
        assert isinstance(content, str)
        parsed = m3u8.loads(content)

        # If we accidentally got a master, pick the first media playlist (existing behavior),
        # and IMPORTANT: update base_url for urljoin to the *new* playlist URL.
        base_url = playlist_url
        if parsed.is_variant:
            self.logger.warning("Media playlist expected; got variant. Resolving to first sub-playlist...")
            media_rel = parsed.playlists[0].uri
            media_url = urljoin(playlist_url, media_rel)
            self.logger.info(f"Resolved to new URL: {media_url}")
            content = await self.fetch(url=media_url, save_cache=False)
            assert isinstance(content, str)
            parsed = m3u8.loads(content)
            base_url = media_url

        segments: List[str] = []

        # Robust init segment handling (EXT-X-MAP)
        # Older m3u8 lib: .segment_map; newer: .init_section
        init_url = None
        segmap = getattr(parsed, "segment_map", None)
        if segmap:
            try:
                init_url = urljoin(base_url, segmap[0].uri)
            except Exception:
                pass
        if init_url is None:
            init_section = getattr(parsed, "init_section", None)
            if init_section and getattr(init_section, "uri", None):
                init_url = urljoin(base_url, init_section.uri)

        if init_url:
            segments.append(init_url)
            self.logger.debug(f"Found init segment: {init_url}")

        # Build absolute URLs for all media segments
        for seg in parsed.segments:
            segments.append(urljoin(base_url, seg.uri))

        self.logger.debug(f"Fetched {len(segments)} segments from m3u8 URL (including init if present)")
        self.logger.info(f"Saving segments to cache....")
        self.cache.save_segments_to_cache(_cache_url, segments)
        return segments

    def _segment_index_width(self, total: int) -> int:
        return max(6, len(str(max(0, total - 1))))

    def _segment_file_path(self, segment_dir, index: int, width: int) -> str:
        return os.path.join(segment_dir, f"seg_{index:0{width}d}.ts")

    def _safe_remove(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug(f"Failed to remove file {path}: {e}")

    def _safe_rmtree(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug(f"Failed to remove directory {path}: {e}")

    def _write_segment_state(self, state_path: str, state: DownloadState) -> None:
        tmp_path = f"{state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(state, fp, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp_path, state_path)

    def _load_segment_state(self, state_path: str) -> Dict[str, Any]:
        with open(state_path, "r", encoding="utf-8") as fp:
            return cast(Dict[str, Any], json.load(fp))

    def _build_segment_state(
        self,
        *,
        segments: List[str],
        missing: List[int],
        segment_dir: Optional[str],
        segment_index_width: int,
        path: str,
        quality: str,
        start_segment: int,
        m3u8_url: Optional[str],
        created_at: Optional[str] = None
    ) -> DownloadState:
        now = datetime.now(timezone.utc).isoformat()
        state = DownloadState(
            version=1,
            created_at=created_at or now,
            updated_at=None,
            m3u8_url=m3u8_url,
            quality=quality,
            output_path=path,
            segment_dir=segment_dir,
            segment_index_width=segment_index_width,
            start_segment=start_segment,
            total=len(segments),
            missing=missing,
            segments=segments
        )
        return state

    async def download_segment(self, url: str, timeout: int, stop_event: Optional[threading.Event] = None) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment.
        Returns (url, content, success).
        """
        try:
            if stop_event is not None and stop_event.is_set():
                return url, b"", False # Stopping the download here

            content = await self.fetch(url, timeout=timeout, get_bytes=True, save_cache=False)
            assert isinstance(content, bytes)
            return url, content, True
        except Exception as e:
            # Log and mark failure; the caller will decide whether to retry or abort.
            self.logger.warning(f"Segment download failed: {url} -> {e}")
            return url, b"", False

    async def download(
        self,
        video: Any, # The video object
        quality: str, # Selected quality e.g., 720, 1080 and so on
        path: str, # Output Path
        callback: Optional[Callable[[int, int], None]] = None, # The callback (function that accepts pos and total)
        remux: bool = False, # Whether to remux the video from MPEG-TS to mp4 container
        callback_remux: Optional[Callable[[int, int], None]] = None, # Callback for the remuxing process
        max_workers_download: int = 20, # The maximum amount of workers that fetch segments at the same time
        start_segment: int = 0, # The start segment to work on (only relevant for resuming)
        stop_event: Optional[threading.Event] = None, # Stop event to exit the download
        segment_state_path: Optional[str] = None, # The path for the segment state (json), for resuming
        segment_dir: Optional[str] = None, # The directory of segments
        return_report: bool = False, # Whether to do a report
        cleanup_on_stop: bool = True,
        keep_segment_dir: bool = False,
        ios_support: bool = False
    ) -> Optional[Union[Dict[str, Any], bool]]:
        """
        :param video:
        :param callback:
        :param quality:
        :param path:
        :param remux:
        :param callback_remux:
        :param max_workers_download:
        :param start_segment:
        :param stop_event:
        :param segment_state_path:
        :param segment_dir:
        :param return_report:
        :param cleanup_on_stop:
        :param keep_segment_dir:
        :param ios_support:
        :return:
        """
        requested_workers = max_workers_download
        max_workers_download = max_workers_download or self.config.max_workers_download # Get max workers from config (fallback)
        if not requested_workers:
            self.logger.debug(f"download: using config max_workers_download={max_workers_download}")

        if callback is None:
            # Use a terminal text progressbar by default
            callback = Callback.text_progress_bar
            self.logger.debug("download: no callback provided, using default text progress bar")

        m3u8_url = getattr(video, "m3u8_base_url", None)
        self.logger.info(
            f"Download requested: quality={quality} path={path} remux={remux} max_workers={max_workers_download} "
            f"start_segment={start_segment} segment_state_path={segment_state_path} segment_dir={segment_dir} "
            f"return_report={return_report} cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir} "
            f"stop_event_set={bool(stop_event and stop_event.is_set())}"
        )
        if m3u8_url:
            self.logger.debug(f"Download m3u8_base_url={m3u8_url}")

        threaded_download = self.threaded(max_workers=max_workers_download, timeout=self.config.timeout)
        self.logger.debug(f"download: dispatching to threaded downloader (timeout={self.config.timeout})")
        return await threaded_download(
            self,
            video=video,
            quality=quality,
            path=path,
            callback=callback,
            remux=remux,
            callback_remux=callback_remux,
            start_segment=start_segment,
            stop_event=stop_event,
            segment_state_path=segment_state_path,
            segment_dir=segment_dir,
            return_report=return_report,
            cleanup_on_stop=cleanup_on_stop,
            keep_segment_dir=keep_segment_dir,
            ios_support=ios_support
        )

        # I removed ffmpeg and default download because you can configure the treaded downloader to behave exactly as the
        # default download by setting max workers to one and ffmpeg as not used by Porn Fetch anymore and I don't wanna
        # say to much, but I think that this implementation is way faster and more **reliable** then FFmpeg. (Yeah PyCharm no shit this code is unreachable, bro it's not even code lol) 
        #
        # *hopefully

    def threaded(self, max_workers: int, timeout: int) -> Callable[..., Coroutine[Any, Any, Optional[Union[Dict[str, Any], bool]]]]:
        # ChatGPT cooked so hard, not even wannabe influencers in dubai get that hot when they get exposed to the sun
        # <xD emoji from Hyprland Discord server here>

        async def wrapper(
            self: "BaseCore",
            video: Any,
            quality: str,
            callback: Optional[Callable[[int, int], None]],
            path: str,
            remux: bool = True,
            callback_remux: Optional[Callable[[int, int], None]] = None,
            start_segment: int = 0,
            stop_event: Optional[threading.Event] = None,
            segment_state_path: Optional[str] = None,
            segment_dir: Optional[str] = None,
            return_report: bool = False,
            cleanup_on_stop: bool = True,
            keep_segment_dir: bool = False,
            ios_support: bool = False
        ) -> Optional[Union[Dict[str, Any], bool]]:
            """
            Threaded HLS segment downloader with optional resume state and stop flag.
            """
            # Cast these to satisfy type checker if they come in as Optional
            cleanup_on_stop = bool(cleanup_on_stop)
            keep_segment_dir = bool(keep_segment_dir)

            self.logger.info(
                f"Threaded download start: quality={quality} path={path} remux={remux} start_segment={start_segment} "
                f"segment_state_path={segment_state_path} segment_dir={segment_dir} return_report={return_report} "
                f"cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir} max_workers={max_workers} "
                f"timeout={timeout} stop_event_set={bool(stop_event and stop_event.is_set())}"
            )
            self.logger.debug(
                f"Threaded download callbacks: callback_set={bool(callback)} callback_remux_set={bool(callback_remux)}"
            )
            resume_state = None
            resume_mode = False
            created_at = None
            
            # Help type checker with initial types
            segments: List[str] = []
            m3u8_url: str = ""

            if segment_state_path:
                if os.path.exists(segment_state_path):
                    self.logger.info(f"Found segment state file: {segment_state_path}. Attempting resume.")
                else:
                    self.logger.debug(f"No segment state file found at: {segment_state_path}. Starting fresh.")

            if segment_state_path and os.path.exists(segment_state_path):
                try: # This starts resuming from previous download
                    resume_state = self._load_segment_state(segment_state_path)
                    resume_mode = True
                except Exception as e: # Shouldn't happen, but if it does, we just do a new download
                    self.logger.warning(f"Failed to load segment state {segment_state_path}: {e}. Starting fresh.")
                    resume_state = None
                    resume_mode = False

            if resume_mode:
                assert resume_state is not None
                segments = cast(List[str], resume_state.get("segments") or []) # This fetches the list of segments from the resume state
                if not segments:
                    raise UnknownError("Segment state is invalid or empty.") # Shouldn't happen ;)

                segment_dir = resume_state.get("segment_dir") or segment_dir
                if not segment_dir:
                    raise UnknownError("Segment state is missing segment_dir.")

                created_at = resume_state.get("created_at")
                width = int(resume_state.get("segment_index_width") or self._segment_index_width(len(segments)))
                state_start = int(resume_state.get("start_segment", 0) or 0) # Where we start writing segments

                """
                Because every segment has a different binary offset, we can't just inject specific segments into specific
                parts of the file. That's why I can only start after xx successful segments.
                
                So, let's say 0-12 segments were successful, but 13 was not and from 14-17 everything went smooth.
                In this case, I need to start from 13 and STILL override 14-17.
                """

                if start_segment and state_start != start_segment:
                    self.logger.warning(
                        f"start_segment={start_segment} ignored; resuming from state start_segment={state_start}."
                    )

                start_segment = state_start
                m3u8_url = cast(str, resume_state.get("m3u8_url") or "")
                state_quality = resume_state.get("quality", quality)
                self.logger.info(
                    f"Resume state loaded: segments={len(segments)} start_segment={start_segment} "
                    f"segment_dir={segment_dir} segment_index_width={width} created_at={created_at} "
                    f"quality={state_quality} m3u8_url={m3u8_url}"
                )

            else:
                m3u8_master = getattr(video, "m3u8_base_url")
                assert m3u8_master is not None, "m3u8_base_url is missing from video object"
                self.logger.info(f"Fetching segments for quality={quality} m3u8_url_master={m3u8_master}")
                segments = await self.get_segments(quality=quality, m3u8_url_master=m3u8_master)
                total_before = len(segments)
                if start_segment > 0:
                    self.logger.debug(
                        f"Applying start_segment offset: {start_segment} (from total={total_before})"
                    )
                    segments = segments[start_segment:]
                if segment_state_path and segment_dir is None:
                    segment_dir = f"{path}.segments"
                    self.logger.debug(f"segment_dir set from state path: {segment_dir}")
                width = self._segment_index_width(len(segments)) if segment_dir else 0
                m3u8_url = m3u8_master
                state_quality = quality
                self.logger.info(
                    f"Segments ready: count={len(segments)} segment_dir={segment_dir} "
                    f"segment_index_width={width} m3u8_url={m3u8_url}"
                )

            n = len(segments) # Total amount of segments
            if n == 0:
                raise UnknownError("No segments found for this playlist.")
                # Shouldn't happen

            if segment_dir:
                os.makedirs(segment_dir, exist_ok=True) # Creates the segment directory for later resuming
                self.logger.debug(f"Segment directory ready: {segment_dir}")
            self.logger.info(f"Segment plan: total={n} segment_dir={segment_dir}")

            downloaded = [False] * n # Keeps track of total downloaded segments

            """
            We write a list with [False, False, n] where n is the value of the total amount of segments.
            This creates a lit with as many False entries as segments. Since `self.download_segment` returns a bool
            along with the data, we can use that to keep track, since we just change the bool to True for every downloaded
            segments.
            """

            if segment_dir: # Tries to find existing segments that we already downloaded
                existing_segments = 0
                for i in range(n): # Does that for every segment
                    seg_path = self._segment_file_path(segment_dir, i, width) # Gets the file path
                    try:
                        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                            # if it exists, we treat it as already downloaded (makes sense)
                            downloaded[i] = True
                            existing_segments += 1
                    except Exception:
                        # If something goes wrong, we treat it as not downloaded and re-fetch it later
                        downloaded[i] = False
                self.logger.info(
                    f"Existing segments detected: {existing_segments}/{n} in {segment_dir}"
                )

            progressed = sum(downloaded) # Amount of already downloaded segments
            downloaded_count = progressed
            if progressed and callback: # Does an initial callback, so that Porn Fetch can start showing the user how
                # many segments have already been downloaded
                callback(progressed, n)
            if progressed:
                self.logger.info(f"Resume progress: already_downloaded={progressed}/{n}")

            target_indices = [i for i in range(n) if not downloaded[i]] # The segments we still need to fetch
            self.logger.info(f"Target segments to download: {len(target_indices)}/{n}")

            tmp_path = f"{path}.tmp" # Creates a temporary path where we write stuff to
            cancelled = False # This is the cancellation event that stops the download
            max_seg_retries = 2 # Maximum retries to get segments
            seg_retries = [0] * n
            progress_log_step = max(1, n // 20)
            next_progress_log = ((progressed // progress_log_step) + 1) * progress_log_step

            if stop_event is not None and stop_event.is_set():
                cancelled = True
                target_indices = [] # Empty list stops the download :)
                self.logger.warning("Stop event already set; cancelling before scheduling segments.")

            if target_indices:
                workers = max(1, min(max_workers, len(target_indices)))
                parts: Optional[List[Optional[bytes]]] = None
                next_to_write = 0
                out_fp = None
                self.logger.info(
                    f"Starting segment download pool: workers={workers} targets={len(target_indices)}"
                )

                if not segment_dir:
                    parts = [None] * n
                    out_fp = cast(Any, open(tmp_path, "wb"))
                    self.logger.debug(f"Using in-memory segment assembly. tmp_path={tmp_path}")
                else:
                    self.logger.debug(f"Writing segments to disk. segment_dir={segment_dir} tmp_path={tmp_path}")

                try:
                    # Use asyncio.gather to fetch segments concurrently instead of ThreadPoolExecutor
                    
                    # Create a semaphore to limit concurrent requests
                    semaphore = asyncio.Semaphore(workers)
                    
                    async def fetch_segment_with_semaphore(idx: int, url: str) -> Tuple[int, bool, bytes]:
                        async with semaphore:
                            if stop_event is not None and stop_event.is_set():
                                return idx, False, b""
                            
                            # Handle retries inside the coroutine
                            for attempt in range(max_seg_retries + 1):
                                if stop_event is not None and stop_event.is_set():
                                    return idx, False, b""
                                    
                                try:
                                    _, data, success = await self.download_segment(url, timeout, stop_event)
                                    if success and data:
                                        return idx, True, data
                                except Exception as e:
                                    self.logger.error(f"Worker exception for segment {idx}: {e}")
                                    
                                if attempt < max_seg_retries:
                                    self.logger.warning(
                                        f"Segment {idx} failed; retrying {attempt + 1}/{max_seg_retries}"
                                    )
                                    # Optional short backoff delay could go here
                                else:
                                    self.logger.error(
                                        f"Segment {idx} failed after {attempt} retries."
                                    )
                            return idx, False, b""
                    
                    tasks = [fetch_segment_with_semaphore(i, segments[i]) for i in target_indices]
                    
                    # Use asyncio.as_completed to process results as they come in, similar to wait(FIRST_COMPLETED)
                    for coro in asyncio.as_completed(tasks):
                        if stop_event is not None and stop_event.is_set():
                            cancelled = True
                            # The remaining tasks will see the event set and exit quickly
                            continue
                            
                        i, success, data = await coro
                        
                        if cancelled:
                            continue

                        if success and data:
                            downloaded[i] = True # Successfully got segment, mark it as done
                            downloaded_count += 1
                            if segment_dir:
                                # Write to a temp path (good for resuming, but not I/O efficient)
                                seg_path = self._segment_file_path(segment_dir, i, width)
                                tmp_seg = f"{seg_path}.part"
                                # Offload segment file writing to a thread
                                def write_part(ts_path: str, t_data: bytes) -> None:
                                    with open(ts_path, "wb") as f:
                                        f.write(t_data)
                                await asyncio.to_thread(write_part, tmp_seg, data)
                                os.replace(tmp_seg, seg_path)
                            else:
                                assert parts is not None
                                parts[i] = data # Keep in memory (I/O efficient)

                            progressed += 1 # Fetched +1 segment, so we give back callback
                            if callback:
                                callback(progressed, n)
                            if progressed >= next_progress_log or progressed == n:
                                remaining = n - downloaded_count
                                self.logger.debug(
                                    f"Segment progress: processed={progressed}/{n} "
                                    f"downloaded={downloaded_count} remaining={remaining}"
                                )
                                next_progress_log += progress_log_step

                        else:
                            # Handling failure (already retried in fetch_segment_with_semaphore)
                            progressed += 1
                            if callback:
                                callback(progressed, n)
                            if progressed >= next_progress_log or progressed == n:
                                remaining = n - downloaded_count
                                self.logger.debug(
                                    f"Segment progress: processed={progressed}/{n} "
                                    f"downloaded={downloaded_count} remaining={remaining}"
                                )
                                next_progress_log += progress_log_step

                        if not segment_dir and parts is not None:
                            chunks_to_write = []
                            while next_to_write < n and parts[next_to_write] is not None:
                                if parts[next_to_write]:
                                    chunks_to_write.append(parts[next_to_write])
                                next_to_write += 1
                            if chunks_to_write:
                                # Write memory chunks to thread to prevent IO block
                                def write_chunks(fp: Any, list_of_data: List[bytes]) -> None:
                                    for c_data in list_of_data:
                                        fp.write(c_data)
                                await asyncio.to_thread(write_chunks, cast(Any, out_fp), cast(List[bytes], chunks_to_write))

                finally:
                    if out_fp is not None:
                        out_fp.close()

            missing = [i for i, ok in enumerate(downloaded) if not ok] # Missing segments
            missing_urls = [segments[i] for i in missing] # Missing URLs of segments
            self.logger.info(
                f"Segment download finished: downloaded={downloaded_count}/{n} missing={len(missing)} cancelled={cancelled}"
            )
            if missing:
                sample = missing[:10]
                self.logger.error(
                    f"Missing segments detected: count={len(missing)} sample={sample}"
                )

            report = DownloadReport(
                status= "cancelled" if cancelled else ("failed" if missing else "completed"),
                total=n,
                downloaded= n - len(missing),
                missing=missing,
                missing_urls=missing_urls,
                segment_dir=segment_dir,
                segment_state_path=segment_state_path,
                start_segment=start_segment,
                quality=quality

            )

            if cancelled: # If user cancels, we clean up stuff
                self.logger.warning(
                    f"Download cancelled. cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir}"
                )
                if cleanup_on_stop:
                    self._safe_remove(tmp_path)
                    if segment_dir and not keep_segment_dir:
                        self._safe_rmtree(segment_dir)

                if segment_state_path:
                    # This is the segment state that is saved as a file, this is NOT the returned report!

                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = self._build_segment_state(
                        segments=segments,
                        missing=missing,
                        segment_dir=segment_dir,
                        segment_index_width=width if segment_dir else 0,
                        path=path,
                        quality=str(state_quality),
                        start_segment=start_segment,
                        m3u8_url=m3u8_url,
                        created_at=created_at,
                    )
                    self._write_segment_state(segment_state_path, state)

                if return_report:
                    missing = report.missing
                    self.logger.debug(
                        f"Returning cancelled report: downloaded={report.downloaded} missing={len(missing)}"
                    )
                    return report
                raise DownloadCancelled("Download cancelled.")

            if missing:
                self.logger.error(
                    f"Download incomplete: {len(missing)} segments missing. Writing state={bool(segment_state_path)}"
                )
                self._safe_remove(tmp_path)
                if segment_state_path:
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = self._build_segment_state(
                        segments=segments,
                        missing=missing,
                        segment_dir=segment_dir,
                        segment_index_width=width if segment_dir else 0,
                        path=path,
                        quality=str(state_quality),
                        start_segment=start_segment,
                        m3u8_url=m3u8_url,
                        created_at=created_at,
                    )
                    self._write_segment_state(segment_state_path, state)
                if return_report:
                    self.logger.debug(
                        f"Returning failed report: downloaded={report.downloaded} missing={len(report.missing)}"
                    )
                    return report
                raise UnknownError(
                    f"Failed to download {len(missing)} segments. Try lowering workers or switching downloader=FFMPEG."
                )

            if segment_dir:
                self.logger.info(
                    f"Assembling {n} segments from {segment_dir} into {tmp_path}"
                )
                def assemble_segments() -> List[int]:
                    with open(tmp_path, "wb") as out_fp:
                        for i in range(n):
                            seg_path = self._segment_file_path(segment_dir, i, width)
                            if not os.path.exists(seg_path):
                                return [i]
                            with open(seg_path, "rb") as seg_fp:
                                shutil.copyfileobj(seg_fp, out_fp, length=1024 * 1024)
                    return []
                
                # Offload heavy IO segment assembly
                missing_assemble = await asyncio.to_thread(assemble_segments)
                if missing_assemble:
                    missing = missing_assemble

                if missing:
                    self.logger.error(
                        f"Missing segment file during assemble: index={missing[0]} segment_dir={segment_dir}"
                    )
                    self._safe_remove(tmp_path)
                    if segment_state_path:
                        self.logger.info(f"Writing segment state to: {segment_state_path}")
                        state = self._build_segment_state(
                            segments=segments,
                            missing=missing,
                            segment_dir=segment_dir,
                            segment_index_width=width if segment_dir else 0,
                            path=path,
                            quality=str(state_quality),
                            start_segment=start_segment,
                            m3u8_url=m3u8_url,
                            created_at=created_at,
                        )
                        self._write_segment_state(segment_state_path, state)
                    report.status = "failed"
                    report.missing = missing
                    report.missing_urls = [segments[i] for i in missing]
                    if return_report:
                        self.logger.debug(
                            f"Returning failed report after assemble: downloaded={report.downloaded} "
                            f"missing={len(report.missing)}"
                        )
                        return report
                    raise UnknownError("Missing segment files during assemble.")

            if remux:
                self.logger.info(f"Remuxing TS to MP4: input={tmp_path} output={path}")
                # Offload heavy CPU/IO bound task
                await asyncio.to_thread(self._convert_ts_to_mp4, tmp_path, path, callback_remux, ios_support)
                # This is important, because not all players can play MPEG-TS AND I want to write
                # metadata to the files, and this doesn't work without a container.
                self._safe_remove(tmp_path)
                self.logger.info(f"Remux completed: output={path}")

            else:
                self.logger.debug("Remux disabled; moving temporary file into place.")
                try:
                    os.replace(tmp_path, path) # If we don't remux, we just rename it to mp4 and treat it as done :)
                except Exception: # Shouldn't happen and I also don't know what this does lol
                    self.logger.warning("os.replace failed; falling back to manual copy.")
                    def manual_copy() -> None:
                        with open(path, "wb") as final_fp, open(tmp_path, "rb") as in_fp:
                            for chunk in iter(lambda: in_fp.read(1024 * 1024), b""):
                                final_fp.write(chunk)
                    await asyncio.to_thread(manual_copy)
                    self._safe_remove(tmp_path) # Remove stuff I guess

            if segment_dir and not keep_segment_dir:
                self._safe_rmtree(segment_dir) # Delete segment dir (cleanup) (optional)
            if segment_state_path: # Delete segment state (optional)
                self._safe_remove(segment_state_path)
            self.logger.info(f"Download completed successfully: path={path}")

            if return_report: # Do a report, if user asked to
                self.logger.debug(
                    f"Returning completed report: downloaded={report.downloaded} missing={len(report.missing)}"
                )
                return report
            return True

        return wrapper # Return the wraper (start stuff)

    def _convert_ts_to_mp4(self, input_path: str, output_path: str, callback: Optional[Callable[[int, int], None]] = None, ios_support: bool = False) -> None:
        start_ts = time.perf_counter()
        self.logger.info(f"Remux start: input={input_path} output={output_path}")
        try:
            input_size = os.path.getsize(input_path)
            self.logger.debug(f"Remux input size: {input_size} bytes")
        except Exception as e:
            self.logger.debug(f"Remux input size unavailable: {e}")

        try:
            from av import open as av_open  # type: ignore[import-not-found]
            from av.audio.resampler import AudioResampler  # type: ignore[import-not-found]

        except (ModuleNotFoundError, ImportError) as e:
            self.logger.error(f"PyAV import failed for remux: {e}")
            raise ModuleNotFoundError(f"PyAV is required for remuxing. Install with pip install av. Not supported on Termux! {e}")

        self.logger.debug(f"Opening input for remux: {input_path}")
        input_ = av_open(input_path)
        fmt_name = (input_.format.name or "").lower()
        self.logger.info(f"Input format detected: {fmt_name or '<unknown>'}")
        if fmt_name == "mpegts":
            output = av_open(output_path, mode="w", format="mp4", options={"movflags": "faststart"})

            # --- VIDEO: keep your exact remux approach ---
            in_video = input_.streams.video[0]
            out_video = output.add_stream_from_template(template=in_video)
            self.logger.debug(
                f"Video stream: codec={getattr(in_video.codec_context, 'name', None)} "
                f"bit_rate={getattr(in_video.codec_context, 'bit_rate', None)}"
            )

            # --- AUDIO: copy if MP4-compatible; else transcode to AAC ---
            in_audio = next((s for s in input_.streams if s.type == "audio"), None)
            out_audio = None
            transcode_audio = False
            resampler = None

            if in_audio:
                # Common MP4-safe audio codecs to copy without transcoding.
                copy_ok = {"aac"} if ios_support else {"aac", "alac", "mp3"}
                codec_name = (in_audio.codec_context.name or "").lower()
                sample_rate = in_audio.codec_context.sample_rate or 0
                layout_name = (
                    in_audio.codec_context.layout.name
                    if getattr(in_audio.codec_context, "layout", None)
                    else "unknown"
                )
                self.logger.debug(
                    f"Audio stream: codec={codec_name} sample_rate={sample_rate} layout={layout_name}"
                )

                if codec_name in copy_ok:
                    out_audio = output.add_stream_from_template(template=in_audio)
                    self.logger.info("Audio codec MP4-compatible; remuxing without transcoding.")
                else:
                    # Build an AAC encoder stream. Match input rate/layout where possible.
                    transcode_audio = True
                    sample_rate = in_audio.codec_context.sample_rate or 48000
                    layout = (
                        in_audio.codec_context.layout.name
                        if getattr(in_audio.codec_context, "layout", None)
                        else "stereo"
                    )
                    out_audio = output.add_stream("aac", rate=sample_rate)
                    self.logger.info(
                        f"Transcoding audio to AAC: sample_rate={sample_rate} layout={layout}"
                    )
                    # Setting layout/channels helps encoder pick sensible defaults.
                    try:
                        out_audio.layout = layout
                    except Exception:
                        pass

                    # Ensure frames handed to the AAC encoder are in a compatible sample format.
                    # ('fltp' is the usual format for AAC encoders.)
                    resampler = AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            else:
                self.logger.info("No audio stream detected; remuxing video only.")

            # --- DEMUX both streams so audio is included ---
            demux_streams = [in_video] + ([in_audio] if in_audio else [])
            packets = input_.demux(demux_streams)  # keeps your progress logic but uses a generator

            try:
                total = os.path.getsize(input_path)
            except Exception:
                total = 100

            self.logger.info(f"Demuxing packets: total_bytes={total}")
            progress_step = max(1, total // 10) if total else 0
            next_progress_log = progress_step if progress_step else 0

            current_progress = 0

            for idx, packet in enumerate(packets):
                pkt_size = getattr(packet, "size", 0) or 0
                current_progress += pkt_size

                if packet.dts is None:
                    if callback:
                        callback(current_progress, total)
                    continue

                if packet.stream == in_video:
                    # Your original video remux path (no explicit rescale).
                    packet.stream = out_video
                    output.mux(packet)

                elif in_audio and packet.stream == in_audio:
                    if not transcode_audio:
                        # Audio is already MP4-compatible; just remux.
                        packet.stream = out_audio
                        output.mux(packet)
                    else:
                        assert out_audio is not None
                        # Decode -> (optionally resample) -> encode AAC -> mux.
                        for frame in packet.decode():
                            # Resample to match encoder expectations.
                            frames = resampler.resample(frame) if resampler else [frame]
                            for f in frames:
                                for enc_pkt in out_audio.encode(f):
                                    output.mux(enc_pkt)

                if callback:
                    callback(current_progress, total)
                if progress_step and current_progress >= next_progress_log:
                    self.logger.debug(f"Remux progress: bytes={current_progress}/{total}")
                    next_progress_log += progress_step

            # Flush audio encoder if we transcoded.
            if transcode_audio and out_audio:
                self.logger.debug("Flushing AAC encoder.")
                for enc_pkt in out_audio.encode(None):
                    output.mux(enc_pkt)

            input_.close()
            output.close()
            elapsed = time.perf_counter() - start_ts
            try:
                out_size = os.path.getsize(output_path)
                self.logger.info(
                    f"Remux complete: output={output_path} size={out_size} bytes elapsed={elapsed:.2f}s"
                )
            except Exception as e:
                self.logger.info(
                    f"Remux complete: output={output_path} elapsed={elapsed:.2f}s (size unavailable: {e})"
                )

        else:
            self.logger.info("Stream seems to be already in MP4! Skipping remux...")
            os.rename(input_path, output_path)
            elapsed = time.perf_counter() - start_ts
            self.logger.info(f"Remux skipped; file moved. elapsed={elapsed:.2f}s")

    async def legacy_download(self, path: str, url: str, callback: Optional[Callable[[int, int], None]] = None,
                        max_retries: int = 5,
                        chunk_size: int = 1024,
                        read_timeout: float = 120.0,
                        stop_event: Optional[threading.Event] = None,
                        max_workers: int = 5,
                        allow_multipart: bool = True) -> bool:
        """
        Download a file using streaming with stall tolerance and resume.
        Supports fast concurrent range downloading if the server supports it and allow_multipart is True.
        Assumes self.session is an AsyncSession.
        """
        self.logger.info(
            f"Legacy download start: url={url} path={path}"
            f"max_retries={max_retries} read_timeout={read_timeout} "
            f"stop_event_set={bool(stop_event and stop_event.is_set())} "
            f"allow_multipart={allow_multipart}"
        )
        if stop_event is not None and stop_event.is_set():
            self.logger.warning("Stop event already set; cancelling legacy download.")
            raise DownloadCancelled("Download cancelled.")

        # Ensure session is initialized
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        progress_bar = None
        if callback is None:
            progress_bar = Callback()
            self.logger.debug("legacy_download: no callback provided, using default progress bar")

        timeout = read_timeout

        # 1. Check if the server supports Range requests and get file size (if multipart is allowed)
        file_size = 0
        accept_ranges = ""
        
        if allow_multipart:
            # We MUST request uncompressed content for range downloads, otherwise:
            # 1) Content-Length from HEAD reflects the compressed size, not the real file size.
            # 2) Mid-file Range requests on compressed streams cause libcurl error 61
            #    ("incorrect header check") because partial gzip lacks a valid header.
            no_compress = {"Accept-Encoding": "identity"}
            try:
                head_resp = await session.head(url, timeout=timeout, allow_redirects=True, headers=no_compress)
                if head_resp.status_code == 405:  # Method Not Allowed, fallback to streaming GET
                    head_resp_stream = await session.request("GET", url, timeout=timeout, allow_redirects=True, stream=True, headers=no_compress)
                    file_size = int(head_resp_stream.headers.get("Content-Length", 0))
                    accept_ranges = head_resp_stream.headers.get("Accept-Ranges", "")
                else:
                    file_size = int(head_resp.headers.get("Content-Length", 0))
                    accept_ranges = head_resp.headers.get("Accept-Ranges", "")
            except Exception as e:
                self.logger.warning(f"Failed to fetch HEAD info for concurrent check: {e}. Falling back to linear download.")

        # 2. Execute Fast Multipart Download if supported and allowed
        if allow_multipart and file_size > 0 and accept_ranges == "bytes":
            self.logger.info(f"Server supports Range requests. Starting fast multipart download for {file_size} bytes.")
            
            # Pre-allocate file
            def allocate_file() -> None:
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.truncate(file_size)
                elif os.path.getsize(path) != file_size:
                    # File exists but size mismatch, truncate to correct size
                    with open(path, "r+b") as f:
                        f.truncate(file_size)
            await asyncio.to_thread(allocate_file)

            # We will use an array to track progress of chunks
            # A chunk map: {chunk_index: bytes_downloaded}
            chunk_progress = {}
            total_downloaded = [0]  # List to allow modification in inner func
            # Determine chunk sizes based on file size, but keep reasonable bounds
            # For massive files, don't create 10,000 workers.
            target_chunk_size = max(chunk_size, min(10 * 1024 * 1024, file_size // 10)) # Between 1MB and 10MB
            
            semaphore = asyncio.Semaphore(max_workers)
            
            async def download_chunk(start: int, end: int, chunk_idx: int) -> bool:
                nonlocal total_downloaded
                headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
                chunk_progress[chunk_idx] = 0
                
                for attempt in range(max_retries + 1):
                    if stop_event is not None and stop_event.is_set():
                        return False
                        
                    try:
                        async with semaphore:
                            resp = await cast(Any, session).request(
                                "GET", url, headers=headers, timeout=timeout, allow_redirects=True, stream=True
                            )
                            resp.raise_for_status()

                            # Open file once for this chunk download attempt
                            f = await asyncio.to_thread(open, path, "rb+")
                            try:
                                await asyncio.to_thread(f.seek, start + chunk_progress[chunk_idx])
                                async for data in resp.aiter_content():
                                    if stop_event is not None and stop_event.is_set():
                                        return False

                                    await asyncio.to_thread(cast(Any, f).write, data)

                                    data_len = len(data)
                                    chunk_progress[chunk_idx] += data_len
                                    total_downloaded[0] += data_len

                                    if callback:
                                        callback(total_downloaded[0], file_size)
                                    elif progress_bar:
                                        progress_bar.text_progress_bar(downloaded=total_downloaded[0], total=file_size)
                            finally:
                                await asyncio.to_thread(f.close)

                            return True # Chunk success

                    except Exception as e:
                        if attempt < max_retries:
                            self.logger.warning(f"Chunk {chunk_idx} failed (attempt {attempt+1}/{max_retries}): {e}")
                            # Reset progress for this chunk before retry
                            total_downloaded[0] -= chunk_progress[chunk_idx]
                            chunk_progress[chunk_idx] = 0
                            await asyncio.sleep(1 * attempt)
                        else:
                            self.logger.error(f"Chunk {chunk_idx} permanently failed: {e}")
                            return False
                return False

            tasks = []
            chunk_idx = 0
            for start in range(0, file_size, target_chunk_size):
                end = min(start + target_chunk_size - 1, file_size - 1)
                tasks.append(download_chunk(start, end, chunk_idx))
                chunk_idx += 1

            results = await asyncio.gather(*tasks)
            
            if progress_bar:
                # We set it to None instead of del to avoid analyzer confusion about potential unassigned reference later
                progress_bar = None
                
            if stop_event is not None and stop_event.is_set():
                raise DownloadCancelled("Download cancelled.")
                
            if not all(results):
                raise NetworkingError("One or more chunks failed to download completely.")
                
            self.logger.info(f"Fast multipart download complete: path={path}")
            return True

        # 3. Fallback to standard linear streaming download
        if not allow_multipart:
            self.logger.info("allow_multipart=False. Forcing linear streaming download.")
        else:
            self.logger.info("Server does not support Range requests or size is 0. Falling back to linear streaming.")
            
        downloaded_so_far = 0
        attempt = 0
        etag = None

        while True:
            if stop_event is not None and stop_event.is_set():
                self.logger.warning("Stop event set; cancelling legacy download.")
                raise DownloadCancelled("Download cancelled.")
            headers = {}
            if downloaded_so_far:
                headers["Range"] = f"bytes={downloaded_so_far}-"
                
            try:
                response = await cast(Any, session).request(
                    "GET", url, headers=headers, allow_redirects=True, timeout=timeout, stream=True
                )
                if downloaded_so_far and response.status_code == 200:
                    self.logger.warning("Server ignored Range request; restarting download from scratch.")
                    downloaded_so_far = 0
                response.raise_for_status()

                etag_cur = response.headers.get("ETag")
                if etag is None:
                    etag = etag_cur
                elif etag_cur and etag_cur != etag:
                    raise RuntimeError("Remote content changed during download")

                total = None
                cr = response.headers.get("Content-Range")
                if cr and "/" in cr:
                    try: total = int(cr.rsplit("/", 1)[1])
                    except ValueError: pass
                if total is None:
                    try: total = int(response.headers.get("Content-Length", "0")) or None
                    except ValueError: pass

                # Fix fallback if total size is still missing
                if total is None:
                    total = 0

                mode = "ab" if downloaded_so_far else "wb"
                f = await asyncio.to_thread(cast(Any, open), path, mode)
                try:
                    await asyncio.to_thread(f.seek, 0, 2)  # Move to EOF
                    async for chunk in response.aiter_content():
                        if stop_event is not None and stop_event.is_set():
                            raise DownloadCancelled("Download cancelled.")
                        if not chunk:
                            continue
                        await asyncio.to_thread(cast(Any, f).write, chunk)
                        downloaded_so_far += len(chunk)

                        if callback:
                            callback(downloaded_so_far, total)
                        elif progress_bar:
                            progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=total)
                finally:
                    await asyncio.to_thread(f.close)

                if progress_bar:
                    progress_bar = None
                self.logger.info(f"Legacy download complete: bytes={downloaded_so_far} path={path}")
                return True
            except RequestsError as e:
                err_str = str(e).lower()
                if "timeout" in err_str or "read" in err_str:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    backoff = min(2 ** attempt, 30)
                    self.logger.warning(f"Read timeout; retrying {attempt}/{max_retries} in {backoff}s")
                    if stop_event is not None and stop_event.wait(backoff):
                        raise DownloadCancelled("Download cancelled.")
                    else:
                        await asyncio.sleep(backoff)
                    continue
                else:
                    raise NetworkingError(f"Stream for: {url} was closed or failed: {e}")
            except DownloadCancelled:
                raise
            except Exception:
                error = traceback.format_exc()
                raise NetworkingError(f"Unknown error for: {url} -->: {error}")


    def truncate(self, name: str, max_bytes: int = 245) -> str:  # only 245, because we need to append .mp4
        """
        Some websites have titles that are so long (lookint at you missav.ws) that you can't name a file like
        that and thus we need to make sure the file name doesn't exceed the OS limits lol
        """
        encoded = name.encode("utf-8")
        if len(encoded) > max_bytes:
            encoded = encoded[:max_bytes]
            # Ensure not to cut in middle of a UTF-8 sequence
            while encoded[-1] & 0b11000000 == 0b10000000:
                encoded = encoded[:-1]
            return cast(bytes, cast(Any, encoded)).decode("utf-8", errors="ignore")
        return name

    @staticmethod
    def str_to_bool(value: str) -> bool:
        # Some function that I have for some reason idk if this has ever been used lmao
        """
        This function is needed for the ArgumentParser for the CLI version of my APIs. It basically maps the
        booleans for the --no-title option to valid Python boolean values.
        """
        val = value.lower()
        if val in ("true", "1", "yes"):
            return True
        if val in ("false", "0", "no"):
            return False
        raise ValueError(f"Invalid boolean value: {value}")