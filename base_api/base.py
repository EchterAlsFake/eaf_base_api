import re
import os
import sys
import ssl
import uuid
import time
import m3u8
import httpx
import random
import certifi
import logging
import traceback
import threading

from itertools import islice
from collections import deque
from functools import lru_cache
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Iterable

try:
    from modules.errors import *
    from modules.config import config
    from modules.progress_bars import Callback

except (ModuleNotFoundError, ImportError):
    from .modules.errors import *
    from .modules.config import config
    from .modules.progress_bars import Callback

UA_DESKTOP_CHROME = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
loggers = {}
HEIGHT_FROM_URI = re.compile(r'(?<!\d)(\d{3,4})[pP](?!\d)')  # e.g., 1080p, 720P


def _normalize_quality_value(q) -> Union[str, int]:
    if isinstance(q, int):
        return q
    s = str(q).lower().strip()
    if s in {"best", "half", "worst"}:
        return s
    m = re.search(r'(\d{3,4})', s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid quality: {q}")


def _choose_quality_from_list(available: List[str | int], target: Union[str, int]):
    # available like ["240", "360", "480", "720", "1080"]
    av = sorted({int(x) for x in available})
    if isinstance(target, str):
        if target == "best":
            return av[-1]
        if target == "worst":
            return av[0]
        if target == "half":
            return av[len(av) // 2]
        raise ValueError("Invalid label.")
    # numeric: highest ≤ target, else closest
    le = [h for h in av if h <= target]
    if le:
        return le[-1]
    # fallback closest (ties -> higher)
    return min(av, key=lambda h: (abs(h - target), -h))


class ErrorVideo:
    """
    Why?:

    Because when working with ThreadPool I give Video objects back, however, if just one video returns on error it will
    raise the error immediately and thus interrupting the whole ThreadPool and this is just the easiest uncomplexest (does this work in English)
    idk 'unkomplizierteste' Lösung ähhh solution that I can think of (bruh I need to learn English)
    """
    def __init__(self, url: str, err: Exception):
        self.url = url
        self._err = err

    def __getattr__(self, _):
        # Any attribute access surfaces the original error
        raise self._err


def _height_from_variant(variant) -> Optional[int]:
    """Extract height from a variant:
    1) stream_info.resolution (w, h)
    2) URI pattern like .../720p/...
    """
    if getattr(variant, "stream_info", None) and variant.stream_info.resolution:
        _, h = variant.stream_info.resolution  # (w, h)
        return int(h)

    if variant.uri:
        m = HEIGHT_FROM_URI.search(variant.uri)
        if m:
            return int(m.group(1))

    return None

def _is_video_playlist(variant) -> bool:
    """Filter out I-frames/audio-only playlists."""
    # m3u8 lib sometimes sets is_iframe if EXT-X-I-FRAME-STREAM-INF is present.
    if getattr(variant, "is_iframe", False):
        return False

    # If codecs known and contain only audio (mp4a, ac-3, ec-3, etc.)
    codecs = getattr(variant.stream_info, "codecs", None) if getattr(variant, "stream_info", None) else None
    if codecs:
        # very light heuristic: if no video codec substring, probably audio-only.
        # video: avc1, hvc1, hev1, vp9, av01, dvh
        if not any(v in codecs.lower() for v in ("avc1", "hvc1", "hev1", "av01", "vp9", "dvh")):
            return False

    return True

def _collect_variants(master: m3u8.M3U8) -> List[Dict[str, Any]]:
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
    def key_fn(v):
        return (v["height"] or 0, v["bandwidth"])
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

def _normalize_quality(quality: Union[str, int]) -> Union[str, int]:
    """Convert '1080p'->1080, '720'->720, keep labels as-is."""
    if isinstance(quality, int):
        return quality
    q = str(quality).strip().lower()
    if q in {"best", "worst", "half"}:
        return q
    m = re.search(r'(\d{3,4})', q)
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid quality value: {quality!r}")

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
        def diff_key(v):
            return (abs(v["height"] - target), -v["height"], v["bandwidth"], v["frame_rate"])
        return sorted(with_height, key=diff_key)[0]

    # If we have no heights at all, fall back to bandwidth ranking
    return sorted(variants, key=lambda v: v["bandwidth"])[-1]



# Default formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def is_android():
    """Detects if the script is running on an Android device."""
    return "ANDROID_ROOT" in os.environ and "ANDROID_DATA" in os.environ

def get_log_file_path(filename="app.log"):
    """Returns a valid log file path that works on Android and other OS."""
    if is_android():
        return os.path.join(os.environ["HOME"], filename)  # Internal app storage
    return filename  # Default for Linux, Windows, Mac


def send_log_message(ip, port, message):
    """Sends a log message to a remote server via HTTP or HTTPS."""
    try:
        url = f"https://{ip}:{port}/feedback"
        with httpx.Client(timeout=5) as client:
            client.post(url, json={"message": message})
    except Exception as e:
        print(f"Failed to send log to {ip}:{port} - {e}", file=sys.stderr)


class HTTPLogHandler(logging.Handler):
    """Custom log handler that sends logs to a remote HTTP server."""

    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port

    def emit(self, record):
        log_entry = self.format(record)
        send_log_message(self.ip, self.port, log_entry)


def setup_logger(name, log_file=None, level=logging.CRITICAL, http_ip=None, http_port=None):
    """Creates or updates a logger for a specific module."""
    if name in loggers:
        logger = loggers[name]
        logger.setLevel(level)

        file_handler_exists = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        http_handler_exists = any(isinstance(h, HTTPLogHandler) for h in logger.handlers)

        if log_file and not file_handler_exists:
            log_file = get_log_file_path(log_file)
            fh = logging.FileHandler(log_file, mode='a')
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        if http_ip and http_port and not http_handler_exists:
            http_handler = HTTPLogHandler(http_ip, http_port)
            http_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(http_handler)

        return logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)

    if log_file:
        log_file = get_log_file_path(log_file)
        if not hasattr(sys, '_first_run'):
            sys._first_run = True
            file_mode = 'w'
        else:
            file_mode = 'a'
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    if http_ip and http_port:
        http_handler = HTTPLogHandler(http_ip, http_port)
        http_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(http_handler)

    loggers[name] = logger
    return logger


class Cache:
    """
    Caches content from network requests
    """

    def __init__(self, config):
        self.cache_dictionary = {}
        self.lock = threading.Lock()
        self.logger = setup_logger("BASE API - [Cache]", level=logging.CRITICAL)
        self.config = config

    def enable_logging(self, log_file=None, level=logging.DEBUG, log_ip=None, log_port=None):
        """
        Enables logging dynamically for this module.
        """
        self.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)

    def handle_cache(self, url):
        if url is None:
            return None

        with self.lock:
            content = self.cache_dictionary.get(url, None)
            return content

    def save_cache(self, url, content):
        with self.lock:
            if len(self.cache_dictionary.keys()) >= self.config.max_cache_items:
                first_key = next(iter(self.cache_dictionary))
                # Delete the first item
                del self.cache_dictionary[first_key]
                self.logger.info(f"Deleting: {first_key} from cache, due to caching limits...")

            self.cache_dictionary[url] = content

    def save_segments_to_cache(self, m3u8_url: str, segments: list):
        with self.lock:
            self.cache_dictionary[m3u8_url] = segments

    def get_segments_from_cache(self, m3u8_url: str):
        with self.lock:
            segments = self.cache_dictionary.get(m3u8_url, None)
            return segments


class Helper:
    def __init__(
        self,
        core,
        video,
        *,
        logger: Optional[logging.Logger] = None,
        log_name: str = "helper.iterator",
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        http_ip: Optional[str] = None,
        http_port: Optional[int] = None,
        other: Optional[Callable] = None
    ):
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
    def chunked(iterable: Iterable[Any], size: int) -> Iterable[list]:
        """
        This function is used to limit page fetching, so that not all pages are fetched at once.
        Now with trace logs about chunk formation and exhaustion.
        """
        if size <= 0:
            raise ValueError("chunk size must be > 0")
        it = iter(iterable)
        idx = 0
        while True:
            block = list(islice(it, size))
            if not block:
                return
            # Logging kept staticmethod-safe by using root logger to avoid needing self
            logging.getLogger("helper.iterator").debug(
                "chunked: yielding block #%d with %d items", idx, len(block)
            )
            idx += 1
            yield block

    def _get_video(self, url: str):
        return self.Video(url, core=self.core)

    def _other_return(self, url: str):
        """In rare cases e.g., Xhamster we don't always return video objects with the iterator"""
        return self.OtherReturn(url, core=self.core)

    def _make_video_safe(self, url: str):
        # Small helper wrapped with verbose logging
        logger = self.logger
        start = time.perf_counter()
        try:
            v = self.Video(url, core=self.core)
            dur = (time.perf_counter() - start) * 1000
            logger.debug("video_init ok url=%s (%.2f ms)", url, dur)
            return v
        except Exception as e:
            dur = (time.perf_counter() - start) * 1000
            logger.exception("video_init FAILED url=%s (%.2f ms): %s", url, dur, e)
            return ErrorVideo(url, e)

    def iterator(
            self,
            page_urls: List[str] = None,
            extractor: Callable = None,
            pages_concurrency: int = 5,
            videos_concurrency: int = 20,
            other_return: bool = False,
    ):
        """
        Yields Video/ErrorVideo in deterministic (page_idx, vid_idx) order while fetching concurrently.
        Stops scraping pages once a 404 page is encountered (self.core.fetch returns an httpx.Response).
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

        def flush_ready():
            nonlocal next_page_idx, next_video_idx
            flushed = 0
            while True:
                if next_page_idx not in page_counts:
                    if flushed:
                        logger.debug(
                            "[%s] flush_ready paused: next_page=%d unknown count; flushed=%d",
                            run_id, next_page_idx, flushed
                        )
                    return
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
                    return

                item = results.pop(key)
                logger.debug(
                    "[%s] flush_ready yield pidx=%d vidx=%d (remaining_results=%d)",
                    run_id, key[0], key[1], len(results)
                )
                flushed += 1
                next_video_idx += 1
                yield item

        page_iter = iter(enumerate(page_urls))

        with ThreadPoolExecutor(max_workers=pages_concurrency) as page_executor, \
                ThreadPoolExecutor(max_workers=videos_concurrency) as video_executor:

            page_in_flight: Dict[Any, Tuple[int, str]] = {}  # future -> (page_idx, url)
            video_in_flight: Dict[Any, Tuple[int, int]] = {}  # future -> (page_idx, vid_idx)

            # queue of discovered videos we haven't submitted yet
            pending_videos: deque[Tuple[int, int, str]] = deque()

            # helper to respect videos_concurrency strictly
            def schedule_videos():
                scheduled = 0
                while pending_videos and len(video_in_flight) < videos_concurrency:
                    pidx, vid_idx, vurl = pending_videos.popleft()
                    if other_return:
                        vf = video_executor.submit(self._other_return, vurl)
                    else:
                        vf = video_executor.submit(self._make_video_safe, vurl)
                    video_in_flight[vf] = (pidx, vid_idx)
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
                fut = page_executor.submit(self.core.fetch, url)
                page_in_flight[fut] = (pidx, url)
                logger.debug(
                    "[%s] scheduled PAGE pidx=%d url=%s (queue=%d)",
                    run_id, pidx, url, len(page_in_flight)
                )

            # Main loop
            while page_in_flight or video_in_flight or pending_videos:
                # Fill any free video slots before waiting
                if pending_videos and len(video_in_flight) < videos_concurrency:
                    schedule_videos()

                waiting_on = set(page_in_flight.keys()) | set(video_in_flight.keys())
                logger.debug(
                    "[%s] waiting futures page=%d video=%d queued_videos=%d",
                    run_id, len(page_in_flight), len(video_in_flight), len(pending_videos)
                )

                # If nothing in flight but items queued (shouldn't happen often), try to schedule then continue
                if not waiting_on:
                    schedule_videos()
                    if not (page_in_flight or video_in_flight):
                        break
                    waiting_on = set(page_in_flight.keys()) | set(video_in_flight.keys())

                done, _ = wait(waiting_on, return_when=FIRST_COMPLETED)

                for fut in done:
                    # PAGE completed
                    if fut in page_in_flight:
                        pidx, url = page_in_flight.pop(fut)
                        try:
                            html_or_resp = fut.result()
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
                            yield from flush_ready()
                            # schedule next page if available (unless we've already stopped)
                            if not stop_after_404:
                                try:
                                    npidx, nurl = next(page_iter)
                                    nfut = page_executor.submit(self.core.fetch, nurl)
                                    page_in_flight[nfut] = (npidx, nurl)
                                    logger.debug(
                                        "[%s] scheduled NEXT PAGE pidx=%d url=%s after failure",
                                        run_id, npidx, nurl
                                    )
                                except StopIteration:
                                    pass
                            continue

                        # Detect a 404 "page" (httpx.Response returned)
                        is_404 = False
                        try:
                            if isinstance(html_or_resp, httpx.Response) and html_or_resp.status_code == 404:
                                is_404 = True
                        except Exception:
                            # Fallback duck-typing if httpx import fails for some reason
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
                            yield from flush_ready()
                            # Do NOT schedule any more pages
                            continue

                        # If we've already hit a 404 and this page index is after it, ignore this page
                        if stop_after_404 and stop_at_page_idx is not None and pidx > stop_at_page_idx:
                            logger.info(
                                "[%s] ignoring PAGE pidx=%d url=%s (after first 404 at pidx=%d)",
                                run_id, pidx, url, stop_at_page_idx
                            )
                            page_counts[pidx] = 0
                            yield from flush_ready()
                            continue

                        # Normal path: Extract video URLs
                        html = html_or_resp
                        try:
                            video_urls = extractor(html) or []
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

                        yield from flush_ready()

                        # Fetch next page only if we haven't seen a 404
                        if not stop_after_404:
                            try:
                                npidx, nurl = next(page_iter)
                                nfut = page_executor.submit(self.core.fetch, nurl)
                                page_in_flight[nfut] = (npidx, nurl)
                                logger.debug(
                                    "[%s] scheduled NEXT PAGE pidx=%d url=%s (inflight_page=%d)",
                                    run_id, npidx, nurl, len(page_in_flight)
                                )
                            except StopIteration:
                                logger.debug("[%s] no more pages to schedule", run_id)

                    # VIDEO completed
                    elif fut in video_in_flight:
                        pidx, vid_idx = video_in_flight.pop(fut)
                        try:
                            val = fut.result()
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
                        yield from flush_ready()
                        schedule_videos()

            # Final flush (should be no-ops, but logged for completeness)
            logger.debug("[%s] final flush start", run_id)
            yield from flush_ready()
            total_ms = (time.perf_counter() - t0) * 1000
            logger.info("[%s] iterator complete in %.2f ms", run_id, total_ms)

class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(self, config=config):
        self.last_request_time = time.time()
        self.total_requests = 0  # Tracks how many requests have been made
        self.session: Optional[httpx.Client] = None
        self.kill_switch = False
        self.config = config
        self.cache = Cache(self.config)
        self.logger = setup_logger("BASE API - [BaseCore]", log_file=False, level=logging.ERROR)
        self.default_headers = {
            "User-Agent": UA_DESKTOP_CHROME,
            "Accept-Language": self.config.locale,
            "Accept-Encoding": "gzip, deflate, br"
        }

    def enable_logging(self, log_file=None, level=logging.DEBUG, log_ip=None, log_port=None):
        """Enables logging dynamically for this module."""
        self.logger = setup_logger(name="BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)
        self.cache.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)

    def enable_kill_switch(self):
        """This will verify your proxy before every request. If the proxy doesn't actually proxy, raise KillSwitch."""
        self.kill_switch = True

    def check_kill_switch(self):
        proxy_ip = self.config.proxy  # Needs to be a dictionary or URL string per httpx
        pattern = re.compile(
            r'^(?P<scheme>http|socks5)://'
            r'(?:\w+:\w+@)?'  # optional user:pass@
            r'(?P<host>[a-zA-Z0-9.-]+)'
            r':(?P<port>\d{1,5})$'
        )
        match = pattern.match(proxy_ip)
        if match:
            self.logger.info(f"Proxy has valid scheme: {match.group('scheme')} ")
        else:
            self.logger.critical("Proxy is INVALID! Exiting.")
            raise InvalidProxy("The Proxy is invalid, all requests will be aborted, please check your configuration!")

        self.logger.info("Checking if proxy is working...")
        self.logger.info("Doing request to httpbin.org to get your real IP to compare it with another request with activated proxy")

        your_ip = httpx.Client().get("https://httpbin.org/ip").json()["origin"]
        self.logger.info(f"Your IP is: {your_ip}")

        self.logger.info("Doing request with Proxy on...")
        proxy_ip_ = self.fetch("https://httpbin.org/ip", bypass_kill_switch=True, get_response=True).json()["origin"]
        self.logger.info(f"Proxy IP is: {proxy_ip_}")

        if your_ip == proxy_ip_:
            self.logger.critical("IP is the same on both requests... Proxy is not working, exiting!")
            raise KillSwitch("CRITICAL PROXY ERROR, CHECK LOGS!")

    def initialize_session(self):
        ctx = ssl.create_default_context(cafile=certifi.where())
        if not self.config.verify_ssl:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        self.session = httpx.Client(
            proxy=self.config.proxy,
            timeout=self.config.timeout,  # e.g. 5s on good connections
            http2=self.config.use_http2,
            verify=ctx,
            follow_redirects=True,
        )
        # Ensure our defaults are on the session
        self.session.headers.update(self.default_headers)

    def enforce_delay(self):
        """Enforces the specified delay in config.request_delay (only if > 0)."""
        delay = self.config.request_delay
        if delay and delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            self.logger.debug(f"Time since last request: {time_since_last_request:.2f} seconds.")
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                self.logger.debug(f"Enforcing delay of {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _merged_headers(self, override: Optional[Dict[str, str]]) -> None:
        """
        Create request headers from current session headers + optional overrides.
        Overrides win, session headers are the base.
        """
        # Copy to avoid mutating session headers
        self.session.headers.update(override)

    def _merged_cookies(self, override: Optional[Dict[str, str]]) -> None:
        """Same as above, but for cookies"""
        self.session.cookies.update(override)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
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

    def fetch(
        self,
        url: str,
        get_bytes: bool = False,
        timeout: Optional[int] = None,
        get_response: bool = False,
        save_cache: bool = True,
        cookies: Optional[Dict[str, str]] = None,
        allow_redirects: bool = True,
        bypass_kill_switch: bool = False,  # prevents infinite loop
        data: Optional[Dict] = None,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict] = None,
    ) -> Union[bytes, str, httpx.Response]:
        """
        Fetch content with retries, optional caching, proxy support, and bandwidth limiting.

        Returns:
            - httpx.Response if get_response=True
            - bytes            if get_bytes=True
            - str (text)       otherwise

        Raises:
            - ResourceGone, ProxySSLError, KillSwitch, UnknownError
            - httpx.HTTPStatusError on unrecoverable HTTP status (e.g., 4xx/5xx after retries, 403/429 exhausted)
            - httpx.RequestError / Timeout errors may bubble up if not recoverable
        """
        if self.session is None:
            self.initialize_session()

        # Cache (only for text mode)
        cache_hit = self.cache.handle_cache(url)
        if cache_hit is not None and not get_bytes and not get_response:
            self.logger.info(f"Fetched content for: {url} from cache!")
            return cache_hit

        req_timeout = timeout or self.config.timeout
        last_response: Optional[httpx.Response] = None

        max_retries = max(1, int(self.config.max_retries))
        for attempt in range(max_retries):
            # backoff (attempt 0 has no extra sleep)
            if attempt >= 1:
                # capped exponential backoff with jitter
                base = min(5.0, 0.5 * (2 ** attempt))
                jitter = random.random() * 0.25  # 0-250ms
                time.sleep(base + jitter)

            try:
                # Only applies if you explicitly set a delay in config (as you noted)
                self.enforce_delay()

                if self.kill_switch and not bypass_kill_switch:
                    self.check_kill_switch()

                # Recompute headers each attempt so changes (like UA switch) take effect
                self._merged_headers(headers)
                self._merged_cookies(cookies)
                self.logger.debug(f"Using Headers: {self.session.headers}")
                self.logger.debug(f"Using Cookies: {self.session.cookies}")

                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=req_timeout,
                    follow_redirects=allow_redirects,
                    data=data,
                    json=json
                )

                last_response = response
                self.total_requests += 1
                status = response.status_code

                # Fast path
                if status == 200:
                    self.logger.debug(f"Attempt {attempt}: Successfully fetched URL: {url}")

                    if get_response:
                        return response

                    # bandwidth-limited read (optional)
                    if self.config.max_bandwidth_mb is not None and self.config.max_bandwidth_mb >= 0.2:
                        raw_content = bytearray()
                        chunk_size = 64 * 1024  # 64 KB
                        speed_limit = self.config.max_bandwidth_mb * 1024 * 1024
                        min_time_per_chunk = chunk_size / speed_limit
                        start_time = time.time()
                        for chunk in response.iter_bytes(chunk_size=chunk_size):
                            raw_content.extend(chunk)
                            elapsed = time.time() - start_time
                            sleep_time = min_time_per_chunk - elapsed
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                            start_time = time.time()
                        raw_content = bytes(raw_content)

                    else:
                        raw_content = response.content

                    if get_bytes:
                        content = raw_content

                    else:
                        # Prefer server-provided/guessed encoding; fallback to utf-8 then latin-1
                        enc = response.encoding or "utf-8"
                        try:
                            content = raw_content.decode(enc, errors="strict")
                        except UnicodeDecodeError:
                            self.logger.warning(f"Content could not be decoded as {enc} ({url}), decoding in 'latin1' instead!")
                            content = raw_content.decode("latin1", errors="replace")

                        if save_cache:
                            self.logger.debug(f"Saving content of {url} to local cache.")
                            self.cache.save_cache(url, content)

                    return content

                # 403 handling: try a UA switch once, then give up
                if status == 403:
                    if attempt == 0 or attempt == 1:
                        # sleep briefly then switch UA for the next try
                        time.sleep(2)
                        self.session.headers.update({
                            "User-Agent": "AppleWebKit/537.36 (KHTML, like Gecko)"
                        })
                        self.logger.warning(f"Switched User-Agent to: {self.session.headers.get('User-Agent')}")
                        # continue to retry with new UA
                        continue

                    else:
                        # After at least one retry with new UA, fail
                        msg = f"Forbidden (403) after {attempt+1} attempts for URL: {url}"
                        self.logger.error(msg)
                        response.raise_for_status()  # raises HTTPStatusError

                # 404: usually not recoverable, but we try once (if attempt==0) in case of transient CDN
                if status == 404:
                    return response # Immediately returning response, because 404 usually means no more content when searching

                # 410: permanent gone
                if status == 410:
                    self.logger.warning(f"""
The resource: {url} is gone. This can happen because of an expired token. I will try to fix this automatically, by creating
a new session and getting a fresh m3u8 URL. This fix is very experimental and I don't know if it works. """)
                    self.initialize_session()





                # 429: rate limited — respect Retry-After if present, else backoff and retry up to cap
                if status == 429:
                    if attempt == 0 or attempt == 1:
                        self.logger.info("Trying 429 bypass (Initializing new session...)")
                        self.initialize_session()
                        continue

                    wait = self._parse_retry_after(response)
                    if wait is None:
                        # fall back to exponential backoff proportional to attempt
                        wait = min(30.0, 0.5 * (2 ** attempt)) + random.random() * 0.5
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Rate limited (429). Waiting {wait:.2f}s then retrying ({attempt+1}/{max_retries}) for {url}.")
                        time.sleep(wait)
                        continue
                    else:
                        self.logger.error(f"Rate limited (429) after {max_retries} attempts for {url}.")
                        return response

                # 5xx: transient server errors — retry until we run out
                if 500 <= status < 600:
                    self.logger.warning(f"Server error {status} on {url}. Retrying ({attempt+1}/{max_retries})...")
                    try:
                        from fake_useragent import UserAgent
                        ua = UserAgent().random
                        self.logger.warning(f"Changed User Agent to: {ua}")
                        self.session.headers.update({"User-Agent": ua})

                    except ModuleNotFoundError:
                        self.logger.warning("Couldn't change user agent, because you don't have fake-useragent installed. This is NOT an error.")

                    continue

                # Other non-200s: let httpx raise a typed error
                if status != 200:
                    self.logger.info(f"HTTP {status} for {url} (attempt {attempt+1}/{max_retries}).")
                    if attempt < max_retries - 1:
                        continue
                    return response

            except httpx.CloseError:
                self.logger.error(f"Attempt {attempt}: The connection has been unexpectedly closed by: {url}. Retrying...")
                continue

            except (httpx.RequestError, httpx.ConnectError) as e:
                self.logger.error(f"Attempt {attempt}: Request error for URL {url}: {e}")
                if "CERTIFICATE_VERIFY_FAILED" in str(e):
                    raise ProxySSLError("Proxy has an invalid SSL certificate, set 'verify = False' in config")
                # retry unless out of attempts
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying ({attempt+1}/{max_retries}) for URL: {url}")
                    continue
                raise

            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                self.logger.error(
                    f"Attempt {attempt}: Timeout for URL {url}: {e}. "
                    f"Consider increasing the timeout or check your connection."
                )
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying ({attempt+1}/{max_retries}) for URL: {url}")
                    continue
                raise

            except httpx.CookieConflict as e:
                self.logger.error(f"Cookie conflict. Aborting this request. Details: {e}")
                raise UnknownError(f"Cookie conflict during request to {url}: {e}") from e

            except httpx.ProxyError as e:
                self.logger.error(f"Proxy Error for {url}: {e}")
                raise KillSwitch("Proxy error when trying a request, aborting!") from e

            except ResourceGone:
                # propagate as-is
                raise

            except Exception as e:
                # Preserve original exception context
                self.logger.error(
                    f"Attempt {attempt}: Unexpected error for {url}: {e}\n{traceback.format_exc()}"
                )
                raise UnknownError(f"Unexpected error for URL {url}: {e}") from e

        # If we get here, we exhausted retries without returning or raising a httpx status error.
        self.logger.error(f"Failed to fetch URL {url} after {max_retries} attempts.")
        if last_response is not None:
            # Raise a typed error with the last response if we have one.
            try:
                last_response.raise_for_status()
            except httpx.HTTPStatusError as e:
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
    def get_m3u8_by_quality(self, m3u8_url: str, quality: Union[str, int]) -> str:
        """
        Return the media-playlist URL for the requested quality.

        quality:
          - 'best' | 'half' | 'worst'
          - 1080 / '1080' / '1080p' (and similar)
        """
        # Resolve master content
        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)
            self.logger.debug("Resolved inline/custom m3u8 master content.")
            base_for_join = ""  # URIs should be absolute in inline cases; join will handle if relative
        else:
            content = self.fetch(url=m3u8_url)
            master = m3u8.loads(content)
            base_for_join = m3u8_url
            self.logger.debug(f"Resolved m3u8 master: {m3u8_url}")

        if not master.is_variant:
            raise ValueError("Provided URL/content is not a master playlist.")

        variants = _collect_variants(master)
        if not variants:
            raise ValueError("No usable video variants found in master playlist.")

        q = _normalize_quality(quality)
        if isinstance(q, str):  # 'best'/'half'/'worst'
            chosen = _pick_by_label(variants, q)
        else:  # numeric height like 1080, 720, etc.
            chosen = _pick_by_height(variants, q)

        full_url = urljoin(base_for_join or m3u8_url, chosen["uri"])
        return full_url

    def list_available_qualities(self, m3u8_url: str) -> List[int]:
        """
        Inspect the master playlist and return sorted unique heights (e.g., [240, 360, 480, 720, 1080]).
        """
        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)
        else:
            content = self.fetch(url=m3u8_url)
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

    def _to_epoch(self, v: str) -> int:
        n = int(v)
        return n // 1000 if n > 10 ** 12 else n  # handle ms too

    def m3u8_expiry_validto(self, url: str, safety_seconds: int = 60):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items() if v}
        vf = q.get("validfrom")
        vt = q.get("validto")
        if not vt:
            return None  # not this scheme
        start_utc = datetime.fromtimestamp(self._to_epoch(vf), tz=timezone.utc) if vf else None
        expires_utc = datetime.fromtimestamp(self._to_epoch(vt), tz=timezone.utc)
        refresh_utc = expires_utc - timedelta(seconds=safety_seconds)
        secs_left = int((expires_utc - datetime.now(timezone.utc)).total_seconds())
        return {
            "start_utc": start_utc,
            "expires_utc": expires_utc,
            "refresh_utc": refresh_utc,
            "seconds_left": max(secs_left, 0),
        }

    # Example


    def get_segments(self, m3u8_url_master: str, quality: Union[str, int]) -> list:
        _cache_url = f"{m3u8_url_master}{quality}"
        _segments: list | None = self.cache.get_segments_from_cache(_cache_url)
        if _segments:
            self.logger.info(f"Received: {len(_segments)} from cache!")
            return _segments

        # Resolve the quality-specific playlist URL (may still be a master in some edge cases)
        playlist_url = self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        self.logger.debug(f"Trying to fetch segment from m3u8 -> {playlist_url}")

        # M3U8s are volatile → don't cache
        content = self.fetch(url=playlist_url, save_cache=False)
        parsed = m3u8.loads(content)

        # If we accidentally got a master, pick the first media playlist (existing behavior),
        # and IMPORTANT: update base_url for urljoin to the *new* playlist URL.
        base_url = playlist_url
        if parsed.is_variant:
            self.logger.warning("Media playlist expected; got variant. Resolving to first sub-playlist...")
            media_rel = parsed.playlists[0].uri
            media_url = urljoin(playlist_url, media_rel)
            self.logger.info(f"Resolved to new URL: {media_url}")
            content = self.fetch(url=media_url, save_cache=False)
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

    def download_segment(self, url: str, timeout: int) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment.
        Returns (url, content, success).
        """
        try:
            # Segments should not be cached (many unique, large, and short-lived).
            content = self.fetch(url, timeout=timeout, get_bytes=True, save_cache=False)
            return url, content, True
        except Exception as e:
            # Log and mark failure; the caller will decide whether to retry or abort.
            self.logger.warning(f"Segment download failed: {url} -> {e}")
            return url, b"", False

    def download(self, video, quality: str, downloader: str, path: str, callback=None, remux: bool = False,
                 callback_remux=None, max_workers_download: int = 20) -> None:
        """
        :param video:
        :param callback:
        :param downloader:
        :param quality:
        :param path:
        :param remux:
        :param callback_remux:
        :param max_workers_download:
        :return:
        """
        max_workers_download = max_workers_download or self.config.max_workers_download

        if callback is None:
            callback = Callback.text_progress_bar

        if downloader == "default":
            self.default(video=video, quality=quality, path=path, callback=callback, remux=remux, callback_remux=callback_remux)

        elif downloader == "threaded":
            threaded_download = self.threaded(max_workers=max_workers_download, timeout=self.config.timeout)
            threaded_download(self, video=video, quality=quality, path=path, callback=callback, remux=remux,
                              callback_remux=callback_remux)

        elif downloader == "FFMPEG":
            self.FFMPEG(video=video, quality=quality, path=path, callback=callback)

    def threaded(self, max_workers: int, timeout: int):
        def wrapper(self, video, quality: str, callback, path: str, remux: bool = True, callback_remux=None):
            """
            This function has already been optimized a lot and is pretty much perfect in what's possible if we
            don't go asynchronous. I don't want to go async for multiple reasons:
            1) Complexity
            2) I don't want to combine async + Qt
            3) Gain is only 20-30%
            4) The site usually throttles us, before we can achieve the maximum potential
            5) This makes 115MB/s already possible. Why would you go over 1gbit/s?
            """

            segments = self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)
            n = len(segments)
            if n == 0:
                raise UnknownError("No segments found for this playlist.")

            # Cap workers to segment count to avoid idle threads
            workers = max(1, min(max_workers, n))

            # Track which parts are ready; keep bytes in memory only until we flush them to disk
            parts: List[Optional[bytes]] = [None] * n
            next_to_write = 0
            completed = 0
            failures = 0

            # Write combined output incrementally to avoid a giant in-memory join
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as out_fp:
                # Submit all downloads
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    # submit all segments
                    pending = {ex.submit(self.download_segment, url, timeout): i
                               for i, url in enumerate(segments)}

                    seg_retries = [0] * n
                    max_seg_retries = 2

                    parts: List[Optional[bytes]] = [None] * n
                    next_to_write = 0
                    progressed = 0  # how many segments have finished (success or final failure)

                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)

                        for fut in done:
                            i = pending.pop(fut)
                            try:
                                _, data, success = fut.result()
                            except Exception as e:
                                self.logger.error(f"Worker exception for segment {i}: {e}")
                                success, data = False, b""

                            if success and data:
                                parts[i] = data
                                # count immediately: a segment finished successfully
                                progressed += 1
                                callback(progressed, n)
                            else:
                                # retry a couple times
                                if seg_retries[i] < max_seg_retries:
                                    seg_retries[i] += 1
                                    pending[ex.submit(self.download_segment, segments[i], timeout)] = i
                                    continue
                                # give up: mark placeholder and still advance progress
                                parts[i] = b""
                                progressed += 1
                                callback(progressed, n)

                            # now try to flush any contiguous ready parts
                            while next_to_write < n and parts[next_to_write] is not None:
                                if parts[next_to_write]:
                                    out_fp.write(parts[next_to_write])
                                next_to_write += 1

            if failures:
                # Clean temp file to avoid leaving corrupted output
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise UnknownError(
                    f"Failed to download {failures} segments. Try lowering workers or switching downloader=FFMPEG.")

            if remux:
                # Reuse your existing converter (atomic replace on success)
                self._convert_ts_to_mp4(tmp_path, path, callback=callback_remux)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            else:
                # Move tmp to final path atomically
                try:
                    os.replace(tmp_path, path)
                except Exception:
                    # Fallback if replace fails across devices
                    with open(path, "wb") as final_fp, open(tmp_path, "rb") as in_fp:
                        for chunk in iter(lambda: in_fp.read(1024 * 1024), b""):
                            final_fp.write(chunk)
                    os.remove(tmp_path)

        return wrapper

    def default(self, video, quality, callback, path, start: int = 0, remux: bool = True, callback_remux=None) -> bool:
        buffer = b''
        segments = self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)[start:]
        length = len(segments)

        for i, url in enumerate(segments):
            for _ in range(5):
                try:
                    segment = self.fetch(url, get_bytes=True)
                    buffer += segment
                    callback(i + 1, length)
                    break
                except Exception as e:
                    self.logger.error(f"Failed to fetch segment {url}: {e}")
                    continue

        if remux:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, 'wb') as file:
                file.write(buffer)
            self._convert_ts_to_mp4(tmp_path, path, callback=callback_remux)
            os.remove(tmp_path)
        else:
            with open(path, 'wb') as file:
                file.write(buffer)

        return True

    def _convert_ts_to_mp4(self, input_path: str, output_path: str, callback=None):
        try:
            from av import open as av_open
            from av.audio.resampler import AudioResampler
        except (ModuleNotFoundError, ImportError) as e:
            raise ModuleNotFoundError(f"PyAV is required for remuxing. Install with pip install av. Not supported on Termux! {e}")

        input_ = av_open(input_path)
        if input_.format.name.lower() == "mpegts":
            output = av_open(output_path, mode="w", format="mp4", options={"movflags": "faststart"})

            # --- VIDEO: keep your exact remux approach ---
            in_video = input_.streams.video[0]
            out_video = output.add_stream_from_template(template=in_video)

            # --- AUDIO: copy if MP4-compatible; else transcode to AAC ---
            in_audio = next((s for s in input_.streams if s.type == "audio"), None)
            out_audio = None
            transcode_audio = False
            resampler = None

            if in_audio:
                # Common MP4-safe audio codecs to copy without transcoding.
                copy_ok = {"aac", "alac", "mp3"}
                codec_name = (in_audio.codec_context.name or "").lower()

                if codec_name in copy_ok:
                    out_audio = output.add_stream_from_template(template=in_audio)
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
                    # Setting layout/channels helps encoder pick sensible defaults.
                    try:
                        out_audio.layout = layout
                    except Exception:
                        pass

                    # Ensure frames handed to the AAC encoder are in a compatible sample format.
                    # ('fltp' is the usual format for AAC encoders.)
                    resampler = AudioResampler(format="fltp", layout=layout, rate=sample_rate)

                # --- DEMUX both streams so audio is included ---
            demux_streams = [in_video] + ([in_audio] if in_audio else [])
            packets = list(input_.demux(demux_streams))  # keeps your progress logic
            total = len(packets)

            for idx, packet in enumerate(packets):
                if packet.dts is None:
                    if callback:
                        callback(idx + 1, total)
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
                        # Decode -> (optionally resample) -> encode AAC -> mux.
                        for frame in packet.decode():
                            # Resample to match encoder expectations.
                            frames = resampler.resample(frame) if resampler else [frame]
                            for f in frames:
                                for enc_pkt in out_audio.encode(f):
                                    output.mux(enc_pkt)

                if callback:
                    callback(idx + 1, total)

            # Flush audio encoder if we transcoded.
            if transcode_audio and out_audio:
                for enc_pkt in out_audio.encode(None):
                    output.mux(enc_pkt)

            input_.close()
            output.close()

        else:
            self.logger.info("Stream seems to be already in MP4! Skipping remux...")
            os.rename(input_path, output_path)

    def FFMPEG(self, video, quality: str, callback, path: str) -> bool:
        try:
            from ffmpeg_progress_yield import FfmpegProgress

        except (ModuleNotFoundError, ImportError):
            raise ModuleNotFoundError("""
You need to install `ffmpeg-progress-yield` to use the FFmpeg download mode.""")
        base_url = video.m3u8_base_url
        new_segment = self.get_m3u8_by_quality(quality=quality, m3u8_url=base_url)
        url_components = base_url.split('/')
        url_components[-1] = new_segment
        new_url = '/'.join(url_components)

        # Build the command for FFMPEG as a list directly
        command = [
            self.config.FFMPEG_PATH,
            "-i", new_url,  # Input URL
            "-bsf:a", "aac_adtstoasc",
            "-y",  # Overwrite output files without asking
            "-c", "copy",  # Copy streams without re-encoding
            path  # Output file path
        ]

        # Initialize FfmpegProgress and execute the command
        ff = FfmpegProgress(command)
        for progress in ff.run_command_with_progress():
            # Update the callback with the current progress
            callback(int(round(progress)), 100)

            if progress == 100:
                return True

        return False

    def legacy_download(self, path: str, url: str, callback=None,
                        chunk_size: int = 1 << 20,  # 1 MiB
                        max_retries: int = 5,
                        read_timeout: float = 120.0) -> bool:
        """
        Download a file using streaming with stall tolerance and resume.
        Assumes self.session is an httpx.Client.
        """
        try:
            # progress UI fallback
            progress_bar = None
            if callback is None:
                progress_bar = Callback()

            # how many bytes we already have (for resume)
            downloaded_so_far = os.path.getsize(path) if os.path.exists(path) else 0
            etag = None
            attempt = 0

            # prefer a longer read timeout to ride out throttling pauses
            timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=30.0)

            while True:
                headers = {
                }
                if downloaded_so_far:
                    headers["Range"] = f"bytes={downloaded_so_far}-"

                # stream the next segment (or whole file if starting fresh)
                with self.session.stream("GET", url, headers=headers,
                                         follow_redirects=True, timeout=timeout) as response:
                    # If we asked for a Range and got 200, server ignored it: start over.
                    if downloaded_so_far and response.status_code == 200:
                        downloaded_so_far = 0  # restart from scratch
                    response.raise_for_status()

                    # track ETag to detect mid-download content changes
                    etag_cur = response.headers.get("ETag")
                    if etag is None:
                        etag = etag_cur
                    elif etag_cur and etag_cur != etag:
                        raise RuntimeError("Remote content changed during download")

                    total = None
                    # Prefer Content-Range total when resuming; else fall back to Content-Length
                    cr = response.headers.get("Content-Range")  # e.g., "bytes start-end/total"
                    if cr and "/" in cr:
                        try:
                            total = int(cr.rsplit("/", 1)[1])
                        except ValueError:
                            total = None
                    if total is None:
                        try:
                            total = int(response.headers.get("Content-Length", "0")) or None
                        except ValueError:
                            total = None

                    mode = "ab" if downloaded_so_far else "wb"
                    with open(path, mode) as file:
                        try:
                            for chunk in response.iter_bytes(chunk_size=chunk_size):
                                if not chunk:
                                    continue
                                file.write(chunk)
                                downloaded_so_far += len(chunk)

                                # update progress
                                if callback:
                                    callback(downloaded_so_far, total or 0)

                                else:
                                    progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=total or 0)

                            # finished successfully
                            if progress_bar:
                                del progress_bar
                            return True

                        except httpx.ReadTimeout:
                            # stall: back off and retry from current offset
                            attempt += 1
                            if attempt > max_retries:
                                raise
                            time.sleep(min(2 ** attempt, 30))
                            continue  # loop retries with Range  # let him coook!!!

        except httpx.StreamClosed:
            self.logger.error(f"Stream for: {url} was closed. This should not happen...")
            raise NetworkingError(f"Stream for: {url} was closed, if this happens again, please report it!")

        except Exception:
            error = traceback.format_exc()
            self.logger.error(f"Unknown (network) error for: {url}. Please report this! -->: {error}")
            raise NetworkingError(f"Unknown error for: {url}. Please report this! -->: {error}")


    def truncate(self, name: str, max_bytes: int = 245) -> str:  # only 245, because we need to append .mp4
        encoded = name.encode("utf-8")
        if len(encoded) > max_bytes:
            encoded = encoded[:max_bytes]
            # Ensure not to cut in middle of a UTF-8 sequence
            while encoded[-1] & 0b11000000 == 0b10000000:
                encoded = encoded[:-1]
            return encoded.decode("utf-8", errors="ignore")
        return name

    @staticmethod
    def str_to_bool(value):
        """
        This function is needed for the ArgumentParser for the CLI version of my APIs. It basically maps the
        booleans for the --no-title option to valid Python boolean values.
        """
        if value.lower() in ("true", "1", "yes"):
            return True

        elif value.lower() in ("false", "0", "no"):
            return False
