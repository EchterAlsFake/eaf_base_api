from __future__ import annotations
import re
import os
import time
import string
import shutil
import random
import asyncio
import inspect
import logging
import traceback
import threading
from functools import lru_cache
from urllib.parse import urljoin
from curl_cffi import CurlOpt # Used for DNS over HTTPS
from curl_cffi.requests.errors import RequestsError
from curl_cffi.requests import AsyncSession, Response
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type, RetryError
from typing import Union, Callable, Tuple, AsyncGenerator, Coroutine, cast, List, Dict, Any, Awaitable, TYPE_CHECKING


# 1. Standardize on relative imports
from base_api.modules.errors import *
from base_api.modules.type_hints import DownloadReport
from base_api.modules.static_functions import (
    load_segment_state, parse_retry_after, log_precondition_failed,
    write_segment_state, build_segment_state, get_segment_index_width,
    segment_file_path, is_video_playlist, height_from_variant,
    pick_by_height, normalize_quality_value,
    parse_challenge, other_challenge, least_factors,
    collect_variants, pick_by_label
)
from base_api.modules.config import config, RuntimeConfig, DownloadConfigHLS, DownloadConfigRAW
from base_api.modules.progress_bars import Callback
from base_api.modules.logger import setup_logger

# 2. Handle optional dependencies cleanly
try:
    import m3u8
except ImportError:
    m3u8 = None

# 3. Handle specific runtime imports
if TYPE_CHECKING:
    from av.audio.codeccontext import AudioCodecContext
    import m3u8

# The following imports are optional, because they depend on per API and I want to be as memory efficient as possible

try:
    import m3u8
    # Needed for all videos that use HLS streaming. Some do not and use mp4 containers / files instead
except (ModuleNotFoundError, ImportError):
    m3u8 = None  # type: ignore


UA_DESKTOP_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/122.0.0.0 Safari/537.36")

REGEX_CHALLENGE = re.compile(r'var p=(\d+); var s=(\d+);.*?(\d+):1;', re.DOTALL)


class Cache:
    """
    Caches content from network requests
    """

    def __init__(self, configuration: "RuntimeConfig") -> None:
        self.cache_dictionary: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.logger = setup_logger("BASE API - [Cache]", level=logging.CRITICAL)
        self.configuration = configuration

    def enable_logging(self, log_file: str | None = None, level: int = logging.DEBUG,
                       log_ip: str | None = None, log_port: str | int | None = None) -> None:
        """
        Enables logging dynamically for this module.
        """
        self.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level,
                                   http_ip=log_ip, http_port=log_port)

    def handle_cache(self, url: str | None) -> Any:
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
                self.logger.info("Deleting: %s from cache, due to caching limits...", first_key)

            self.cache_dictionary[url] = content

    def save_segments_to_cache(self, m3u8_url: str, segments: List[Any]) -> None:
        with self.lock:
            self.cache_dictionary[m3u8_url] = segments

    def get_segments_from_cache(self, m3u8_url: str) -> List[str] | None:
        with self.lock:
            segments = self.cache_dictionary.get(m3u8_url, None)
            return segments

    def delete_cache(self, entry: str) -> None:
        with self.lock:
            self.cache_dictionary.pop(entry)


class Helper:
    """
    Orchestrates the concurrent fetching and processing of paginated content.

    This class manages the lifecycle of a scraping session, handling multiple pages
    concurrently, extracting video URLs from them, and then fetching video details,
    all while ensuring that the results are yielded in the original order they appeared
    in the pagination.
    """
    def __init__(
        self,
        core: BaseCore,
        video_constructor: Callable[..., Any],
        *,
        logger: logging.Logger | None = None,
        log_name: str = "helper.iterator",
        log_file: str | None = None,
        log_level: int = logging.INFO,
        http_ip: str | None = None,
        http_port: int | str | None = None,
        alternative_constructor: Callable[..., Any] | None = None
    ) -> None:
        """
        Initializes the Scraping Helper.

        Args:
            core: The engine responsible for network requests (must have a .fetch(url) method).
            video_constructor: A factory function/class to create Video objects from a URL and HTML.
            logger: An optional pre-configured logger instance.
            log_name: Name for the logger if one needs to be created.
            log_file: Optional file path to log output to.
            log_level: Logging severity level (e.g., logging.INFO)
            http_ip: Optional IP address for remote logging.
            http_port: Optional port for remote logging.
            alternative_constructor: An optional factory for non-standard results (e.g., when a page doesn't yield videos).
        """
        super().__init__()
        self.core = core
        self.video_factory = video_constructor
        self.alternative_factory = alternative_constructor

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


    async def iterator(
            self,
            target_page_urls: List[str],
            video_link_extractor: Callable[..., Any] | None = None,
            max_page_concurrency: int = 5,
            max_video_concurrency: int = 20,
            use_alternative_constructor: bool = False,
            page_request_method: str = "GET",
            video_request_method: str = "GET",
            ignore_errors: bool = True,
            on_video_error: Callable[[str, Exception, int], Awaitable[bool]] | None = None,
            on_page_error: Callable[[str, Exception, int], Awaitable[bool]] | None = None,
            keep_original_order: bool = False

    ) -> AsyncGenerator[Any, None]:
        """
        The main scraping engine that orchestrates concurrent page and video processing.
        """
        logger = self.logger
        video_queue = asyncio.Queue() # The page worker will give his video URLs into this queue (consumer / producer pattern)
        page_queue = asyncio.Queue() # Stores the page URLs in a queue to apply retry logic later
        results_queue = asyncio.Queue() # This is the queue that stores the actual videos
        page_videos_count: dict[int, int] = {}


        async def fetch_page(url: str) -> str:
            """
            Fetches the HTML content of a page (URL) using a strict number of maximum concurrent tasks as defined
            by the semaphore and 'max_page_concurrency' variable.

            Returns the HTML Content
            """
            logger.debug(f"Fetching Page: {url}")
            return await self.core.fetch(url, method=page_request_method)

        async def fetch_video(url: str) -> tuple[str, str]:
            """
            Fetches the HTML content of a Video....
            """
            logger.debug(f"Fetching Video: {url}")
            html = await self.core.fetch(url, method=video_request_method)
            return url, html

        async def page_worker():
            while True:
                try:
                    page_index, page_url, attempt_count = await page_queue.get()

                except asyncio.CancelledError:
                    return # Exits the loop

                task_cleared = False

                try:
                    self.logger.debug(f"Fetching Page HTML: {page_url}")
                    html_content = await fetch_page(page_url)
                    extracted_videos = await asyncio.to_thread(video_link_extractor, html_content)
                    page_videos_count[page_index] = len(extracted_videos)
                    # When we pass the HTML content to the extractor so that bs4 can extract it, it will take a minimum time
                    # of like 20ms. This would block our event loop and prevent new network requests, so this optimizes the
                    # speed by offloading it to another thread

                except Exception as e:
                    self.logger.error(f"Failed to fetch page: {e}")

                    if on_page_error is not None or ignore_errors:
                        try:
                            if on_page_error is not None:
                                should_retry = await on_page_error(page_url, e, attempt_count)

                                if should_retry:
                                    self.logger.info(f"Re-Queuing {page_url}!")
                                    await page_queue.put((page_index, page_url, attempt_count + 1))
                                    page_queue.task_done()
                                    task_cleared = True
                                    continue

                        except Exception as callback_error:
                            self.logger.error(f"Error inside the provided callback: {callback_error}")
                            raise CallbackError("""
Warning: Your callback did not return True, because of that the failed video will NOT be appended to the queue
and it will be skipped processing!""") from callback_error

                    if not task_cleared:
                        page_videos_count[page_index] = 0

                if page_index in page_videos_count and page_videos_count[page_index] > 0:
                    for video_idx, video_url in enumerate(extracted_videos):
                        await video_queue.put((page_index, video_idx, video_url, 1))

                if not task_cleared:
                    page_queue.task_done()

        async def process_video(video):
            cleaned_video = await video.init()
            return cleaned_video


        async def video_worker():
            while True:
                try:
                    page_index, video_index, video_url, attempt_count = await video_queue.get() # Pulls the Video URL from the queue

                except asyncio.CancelledError:
                    return # Exit the loop if we already canceled remaining parts

                result = ScrapeResult(video_url)
                task_cleared = False
                try:
                    self.logger.debug(f"Fetching Video HTML: {video_url}")
                    url, html = await fetch_video(video_url)
                    if use_alternative_constructor:
                        video_instance = self.alternative_factory(url, core=self.core, html_content=html)

                    else:
                        video_instance = self.video_factory(url, core=self.core, html_content=html) # Creates the video scrape object


                    processed = await process_video(video_instance)
                    await video_instance.clean() # Basically wipes the instance to free up memory
                    result.video = processed
                    result.is_success = True
                    await results_queue.put((page_index, video_index, result))
                    # In this case the video was successfully fetched

                except Exception as e:
                    logger.error(f"Failed to scrape Video URL: {e}")
                    if on_video_error is not None or ignore_errors:
                        try:
                            if on_video_error is not None:
                                should_retry = await on_video_error(video_url, e, attempt_count)

                                if should_retry:
                                    self.logger.info(f"Re-Queuing {video_url}!")
                                    await video_queue.put((page_index, video_index, video_url, attempt_count + 1))
                                    video_queue.task_done()
                                    task_cleared = True
                                    continue

                        except Exception as callback_error:
                            self.logger.error(f"Error inside the provided callback: {callback_error}")
                            raise CallbackError("""
Warning: Your callback did not return True, because of that the failed video will NOT be appended to the queue
and it will be skipped processing!""") from callback_error


                    result.error = e
                    await results_queue.put((page_index, video_index, result))

                finally:
                    if not task_cleared:
                        video_queue.task_done()

        async def create_page_queue():
            for index, url in enumerate(target_page_urls):
                await page_queue.put((index, url, 1))

        async def worker_supervisor():
            await page_queue.join()
            await video_queue.join()
            await results_queue.put(None)

        async with asyncio.TaskGroup() as tg:
            fill_page_queue = tg.create_task(create_page_queue())
            page_tasks = [tg.create_task(page_worker()) for _ in range(max_page_concurrency)]
            video_workers = [tg.create_task(video_worker()) for _ in range(max_video_concurrency)]
            supervisor = tg.create_task(worker_supervisor())

            expected_page = 0
            expected_video = 0
            buffer = {}

            while True:
                result_item = await results_queue.get()
                if result_item is None:
                    for task in page_tasks + video_workers:
                        task.cancel()

                    if keep_original_order:
                        while True:
                            if expected_page in page_videos_count and expected_video >= page_videos_count[expected_page]:
                                expected_page += 1
                                expected_video = 0
                                continue

                            if (expected_page, expected_video) in buffer:
                                yield buffer.pop((expected_page, expected_video))
                                expected_video += 1


                            else:
                                break

                    break

                page_idx, video_idx, result_obj = result_item
                if not keep_original_order:
                    yield result_obj

                else:
                    buffer[(page_idx, video_idx)] = result_obj

                    while True:
                        if expected_page in page_videos_count and expected_video >= page_videos_count[expected_page]:
                            expected_page += 1
                            expected_video = 0
                            continue

                        if (expected_page, expected_video) in buffer:
                            yield buffer.pop((expected_page, expected_video))
                            expected_video += 1

                        else:
                            break



class ScrapeResult:
    def __init__(self, url: str):
        self.url = url
        self.video: Any = None
        self.error: Exception | None = None
        self.is_success: bool = False


class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(self, configuration: "RuntimeConfig" = config) -> None:
        self.lock = asyncio.Lock()
        self.latest_key: str | None = None
        self.latest_key_time: float = 0.0
        self.last_request_time = time.time()
        self.total_requests: int = 0  # Tracks how many requests have been made
        self.session: AsyncSession | None = None
        self.configuration = configuration
        self.cache = Cache(self.configuration)
        self.logger = setup_logger("BASE API - [BaseCore]", log_file=None, level=logging.ERROR)
        self.default_headers = {
            "User-Agent": UA_DESKTOP_CHROME,
            "Accept-Language": self.configuration.locale,
            "Accept-Encoding": "gzip, deflate, br"
        }

    def enable_logging(self, log_file: str | None = None, level: int = logging.DEBUG, log_ip:
    str | None = None, log_port: int | str | None = None) -> None:
        """Enables logging dynamically for this module."""
        self.logger = setup_logger(name="BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip,
                                   http_port=log_port)
        self.cache.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip,
                                         http_port=log_port)

    def initialize_session(self) -> None:
        verify = self.configuration.verify_ssl

        curl_options: Dict[CurlOpt, Union[bytes, int]] = {}
        if self.configuration.dns_over_https:
            curl_options[CurlOpt.DOH_URL] = str(self.configuration.dns_over_https).encode("utf-8")

        proxies = None
        if self.configuration.proxies:
            proxies = self.configuration.proxies

        if self.configuration.max_bandwidth_mb is not None and self.configuration.max_bandwidth_mb > 0:
            global_limit_bytes = int(self.configuration.max_bandwidth_mb * 1024 * 1024)
            total_concurrent_connections = (self.configuration.max_workers_download *
                                            self.configuration.videos_concurrency)
            per_connection_limit = max(1, int(global_limit_bytes / total_concurrent_connections))
            curl_options[CurlOpt.MAX_RECV_SPEED_LARGE] = per_connection_limit

        js3 = self.configuration.custom_ja3
        impersonation = self.configuration.impersonation
        http_version = self.configuration.http_version
        proxy_auth_str = self.configuration.proxy_auth
        trust_env = self.configuration.trust_env

        p_auth: Tuple[str, str] | None = None
        if proxy_auth_str and ":" in proxy_auth_str:
            u, p = proxy_auth_str.split(":", 1)
            p_auth = (u, p)

        self.session = cast(Any, AsyncSession)(
            proxies=proxies,
            timeout=self.configuration.timeout,
            verify=verify,
            impersonate=impersonation,
            curl_options=curl_options,
            http_version=http_version,
            ja3=js3,
            proxy_auth=p_auth,
            trust_env=trust_env
        )
        # Ensure our defaults are on the session
        assert self.session is not None
        self.session.headers.update(self.default_headers)

    async def enforce_delay(self) -> None:
        """Enforces the specified delay in config.request_delay (only if > 0)."""
        delay = self.configuration.request_delay
        if delay and delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            self.logger.debug("Time since last request: {:.2f} seconds.".format(time_since_last_request))
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                self.logger.debug("Enforcing delay of {:.2f} seconds.".format(sleep_time))
                await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()

    def _merged_headers(self, override: Dict[str, str] | None) -> Dict[str, Any]:
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

    def _merged_cookies(self, override: Dict[str, str] | None) -> Dict[str, Any]:
        """Same as above, but for cookies"""
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        cookies: Dict[str, Any] = cast(Dict[str, Any], cast(Any, session.cookies.get_dict()))
        if override:
            cookies.update(override)
        return cookies

    async def fetch(
        self,
        url: str,
        get_bytes: bool = False,
        timeout: int | None = None,
        get_response: bool = False,
        save_cache: bool = True,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> bytes | str | Response:
        """
        Fetch content with retries, optional caching, proxy support, and bandwidth limiting.
        Now uses Tenacity for robust retry logic.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        # Cache (only for text mode)
        cache_hit = self.cache.handle_cache(url)
        if cache_hit is not None and not get_bytes and not get_response:
            self.logger.info("Fetched content for: %s from cache!", url)
            return cache_hit

        req_timeout = timeout or self.configuration.timeout
        max_retries = max(1, int(self.configuration.max_retries))

        # We will use AsyncRetrying for the core retry logic
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=30.0, jitter=0.5),
            retry=retry_if_exception_type((RequestsError, NetworkRequestError)),
            reraise=True
        )

        # Track state that changes across retries
        state = {
            "last_response": None,
            "ua_switched": False
        }

        try:
            async for attempt in retryer:
                with attempt:
                    try:
                        await self.enforce_delay()
                        req_headers = self._merged_headers(headers)
                        if state["ua_switched"]:
                            req_headers["User-Agent"] = "AppleWebKit/537.36 (KHTML, like Gecko)"
                            
                        req_cookies = self._merged_cookies(cookies)
                        
                        if isinstance(self.configuration.max_bandwidth_mb, int):
                            speed_limit = self.configuration.max_bandwidth_mb * 1024 * 1024
                        else:
                            speed_limit = None

                        current_time = asyncio.get_event_loop().time()
                        latest_key = self.latest_key
                        if "KEY" not in session.cookies and latest_key is not None:
                            if current_time - self.latest_key_time < 10:
                                session.cookies.set("KEY", latest_key, domain=".pornhub.com", path="/")

                        response = await cast(Any, session).request(
                            method=cast(Any, method),
                            url=url,
                            timeout=req_timeout,
                            allow_redirects=allow_redirects,
                            data=data,
                            json=json_data,
                            params=params,
                            headers=req_headers,
                            cookies=req_cookies,
                            max_recv_speed=speed_limit or 0,
                        )

                        state["last_response"] = response
                        self.total_requests += 1
                        status = response.status_code

                        content_type = response.headers.get("content-type", "").lower()
                        is_html = "text/html" in content_type if content_type else (not get_bytes)

                        if is_html:
                            enc = getattr(response, "encoding", None) or "utf-8"
                            resp_text = cast(bytes, response.content).decode(enc, errors="replace")

                            if 'onload="go()"' in resp_text:
                                local_latest = getattr(self, "latest_key", None)
                                async with self.lock:
                                    if getattr(self, "latest_key", None) != local_latest:
                                        self.logger.info("Another task already resolved the challenge! Retrying request with the new cookie.")
                                        if self.latest_key:
                                            session.cookies.set("KEY", self.latest_key, domain=".pornhub.com", path="/")

                                        await asyncio.sleep(1.5)
                                        continue

                                    self.logger.info("Challenge page detected! Solving...")
                                    get_challenge = re.compile(r'go\(\).*?{(.*?)n=l.*?KEY.*?s\+":(\d+):', re.DOTALL)
                                    challenge_data = re.search(get_challenge, resp_text)

                                    if challenge_data:
                                        try:
                                            challenge_str, token_str = challenge_data.groups()
                                            code = parse_challenge(challenge_str)
                                            code = other_challenge(code)
                                            code = '\n'.join(code.split(';'))

                                            safe_chars = set(string.ascii_letters + string.digits + " \t\n=+-*/().:><&|~^")
                                            if not all(c in safe_chars for c in code):
                                                self.logger.error("Security Abort: Illegal chars in challenge, CODE: %s", code)
                                                raise SecurityAbort

                                            safe_globals: Dict[str, Any] = {"__builtins__": {}}
                                            safe_locals = {"p": 0, "s": 0}
                                            exec(code, safe_globals, safe_locals)

                                            p = safe_locals.get('p', 0)
                                            s = safe_locals.get('s', 0)
                                            n = least_factors(p)
                                            cookie_value = f'{n}*{p // n}:{s}:{token_str}:1'

                                            self.latest_key = cookie_value
                                            self.latest_key_time = asyncio.get_event_loop().time()
                                            session.cookies.set("KEY", cookie_value, domain=".pornhub.com", path="/")
                                            self.logger.info("RESOLVED CHALLENGE! Injected cookie: %s", cookie_value)

                                            try:
                                                self.cache.delete_cache(url)
                                            except (KeyError, Exception):
                                                pass

                                            await asyncio.sleep(1.5)
                                            continue
                                        except Exception as challenge_error:
                                            raise ChallengeMathError from challenge_error

                                    else:
                                        self.logger.error("Detected challenge page, but the regex failed to extract data.")
                                        await asyncio.sleep(1.5)
                                        raise ChallengeRegexError("Detected Challenge, but regex couldn't extract, report this!")


                        if status == 200:
                            self.logger.debug("Successfully fetched URL: %s", url)
                            if get_response:
                                return response
                            raw_content = response.content
                            if get_bytes:
                                content = raw_content
                            else:
                                enc = getattr(response, "encoding", None) or "utf-8"
                                try:
                                    content = cast(bytes, raw_content).decode(enc, errors="strict")
                                except UnicodeDecodeError:
                                    self.logger.warning("Content could not be decoded as %s (%s), decoding 'latin1' instead!", enc, url)
                                    content = cast(bytes, raw_content).decode("latin1", errors="replace")
                                if save_cache:
                                    self.cache.save_cache(url, content)
                            return content

                        elif status == 204:
                            return response # No content left

                        if status == 403:
                            raise AccessDeniedError("Request blocked by server!")

                        if status == 412:
                            log_precondition_failed(logger=self.logger, attempt=attempt.retry_state.attempt_number, response=response)

                        if status == 404:
                            return response

                        if status == 410:
                            raise ResourceGone(f"Resource gone (HTTP 410) for URL: {url}")

                        if status == 429:
                            wait = parse_retry_after(logger=self.logger, response=response)
                            if wait is not None:
                                self.logger.warning(f"Rate limited (429). Server requested {wait}s pause.")
                                await asyncio.sleep(wait)

                            else:
                                delay = random.randint(2, 6)

                                self.logger.warning(
                                    f"Rate limited (429). No header found. Backing off {delay}s.")
                                await asyncio.sleep(delay)

                            if attempt.retry_state.attempt_number <= 2:
                                raise RateLimitError("429 Rate Limited", retry_after=delay, url=url)

                            continue

                        if status == 401:
                            return response # Expected (for Vinted OSINT script I use)

                        if 500 <= status < 600:
                            self.logger.warning("Server error %s on %s. Retrying...", status, url)
                            raise HTTPStatusError(f"Server error {status}", status_code=status, url=url)

                        if status != 200:
                            self.logger.info("HTTP %s for %s.", status, url)
                            raise NetworkRequestError(f"HTTP {status}")

                    except RequestsError as e:
                        err_str = str(e).lower()
                        self.logger.error("Request error for URL %s: %s", url, e)
                        if "certificate verify failed" in err_str:
                            raise ProxySSLError("Proxy has an invalid SSL certificate, set 'verify = False' in config")
                        elif "cookie conflict" in err_str:
                            raise UnknownError(f"Cookie conflict during request to {url}: {e}") from e
                        elif "proxy" in err_str:
                            raise InvalidProxy("Proxy error when trying a request, aborting!") from e
                        elif "timeout" in err_str or "read" in err_str:
                            self.logger.error("Timeout for URL %s: %s", url, e)
                        raise
                    except (RequestsError, NetworkRequestError, ResourceGone):
                        raise

                    except (SecurityAbort, ProxySSLError, InvalidProxy, UnknownError):
                        raise

                    except Exception as e:
                        self.logger.error("Unexpected error for %s: %s\n%s", url, e, traceback.format_exc())
                        raise UnknownError(f"Unexpected error for URL {url}: {e}") from e

        except RetryError as re_err:
            last_resp = state.get("last_response")
            self.logger.error("Failed to fetch URL %s after %s attempts.", url, max_retries)
            if last_resp is not None:
                try:
                    last_resp.raise_for_status()
                except Exception as e:
                    raise e
            raise UnknownError(
                f"Failed to fetch: {url} after {max_retries} attempts. "
                "If you're sure you're not blocked and your connection is stable, "
                "please open an issue with the URL and steps to reproduce."
            ) from re_err


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

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)
            self.logger.debug("Resolved inline/custom m3u8 master content.")
            base_for_join = ""  # URIs should be absolute in inline cases; join will handle if relative
        else:
            content = await self.fetch(url=m3u8_url)
            assert isinstance(content, str)
            master = m3u8.loads(content)
            base_for_join = m3u8_url
            self.logger.debug("Resolved m3u8 master: %s", m3u8_url)

        if not master.is_variant:
            raise PlaylistExtractionError(f"Playlist is not a master Playlist: {m3u8_url}")

        variants = collect_variants(master)
        if not variants:
            raise PlaylistExtractionError(f"No usable variants found in master Playlist: {m3u8_url}, {master}")

        q = normalize_quality_value(quality)
        if isinstance(q, str):  # 'best'/'half'/'worst'
            chosen = pick_by_label(variants, q)
        else:  # numeric height like 1080, 720, etc.
            chosen = pick_by_height(variants, q)

        full_url = urljoin(base_for_join or m3u8_url, chosen["uri"])
        return full_url

    async def list_available_qualities(self, m3u8_url: str) -> List[int]:
        """
        Inspect the master playlist and return sorted unique heights (e.g., [240, 360, 480, 720, 1080]).
        """
        assert m3u8 is not None

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if not m3u8_url.startswith("https://"):
            master = m3u8.loads(m3u8_url)
        else:
            content = await self.fetch(url=m3u8_url)
            assert isinstance(content, str)
            master = m3u8.loads(content)

        if not master.is_variant:
            return []

        heights = {h for h in (height_from_variant(v) for v in master.playlists) if h is not None}
        if heights:
            return sorted(heights)
        # fallback: bandwidth-only (roughly infer tiers)
        by_bw = sorted(
            (getattr(v.stream_info, "bandwidth", 0) for v in master.playlists if is_video_playlist(v)),
            key=int
        )
        # Return rank numbers instead of heights if we truly can't infer—kept simple:
        return [i for i, _ in enumerate(by_bw, start=1)]

    async def get_segments(self, m3u8_url_master: str, quality: Union[str, int]) -> List[str]:
        assert m3u8 is not None
        _cache_url = f"{m3u8_url_master}{quality}"
        _segments: List[str] | None = self.cache.get_segments_from_cache(_cache_url)
        if _segments is not None:
            self.logger.info("Received: %s from cache!", len(_segments))
            return _segments

        # Resolve the quality-specific playlist URL (may still be a master in some edge cases)
        playlist_url = await self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        self.logger.debug("Trying to fetch segments from m3u8 -> %s", playlist_url)

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
            self.logger.info("Resolved to new URL: %s", media_url)
            content = await self.fetch(url=media_url, save_cache=False)
            assert isinstance(content, str)
            parsed = m3u8.loads(content)
            base_url = media_url

        segments: List[str] = []

        # Robust init segment handling (EXT-X-MAP)
        # Older m3u8 lib: .segment_map; newer: .init_section
        init_url = None
        segments_map = getattr(parsed, "segment_map", None)
        if segments_map:
            assert isinstance(segments_map, list)
            try:
                init_url = urljoin(base_url, segments_map[0].uri)
            except Exception as exc:
                self.logger.info("Couldn't get init url, this is probably not an issue: %s", exc)
                pass
        if init_url is None:
            init_section = getattr(parsed, "init_section", None)
            if init_section and getattr(init_section, "uri", None):
                init_url = urljoin(base_url, init_section.uri)

        if init_url:
            segments.append(init_url)
            self.logger.debug("Found init segment: %s", init_url)

        # Build absolute URLs for all media segments
        for seg in parsed.segments:
            segments.append(urljoin(base_url, seg.uri))

        self.logger.debug("Fetched %s segments from m3u8 URL (including init if present)", len(segments))
        self.logger.info("Saving segments to cache....")
        self.cache.save_segments_to_cache(_cache_url, segments)
        return segments


    def _safe_remove(self, path: str | None) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug("Failed to remove file %s: %s", path, e)

    def _safe_rmtree(self, path: str | None) -> None:
        if not path:
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug("Failed to remove directory %s: %s", path, e)

    async def download_segment(self, url: str, timeout: int, stop_event:
                                threading.Event | None = None) -> tuple[str, bytes, bool]:
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
            self.logger.warning("Segment download failed: %s -> %s", url, e)
            return url, b"", False

    async def download(
        self,
        configuration: DownloadConfigHLS
    ) -> DownloadReport | bool:
        """
        :param video:
        :param configuration:
        :return:
        """

        if configuration.callback is None:
            # Use a terminal text progressbar by default
            configuration.callback = Callback.text_progress_bar
            self.logger.debug("download: no callback provided, using default text progress bar")

        m3u8_url = configuration.m3u8_base_url

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if m3u8_url:
            self.logger.debug("Download m3u8_base_url=%s", m3u8_url)

        self.logger.debug("download: dispatching to threaded downloader (timeout=%s)", self.configuration.timeout)

        # 2. Call the downloader method directly
        return await self.threaded_download(
            configuration=configuration,
            pre_resolved_m3u8=m3u8_url,
            timeout=config.timeout,
            max_workers=config.max_workers_download
        )

    async def threaded_download(
        self: "BaseCore",
        timeout: int,
        max_workers: int,
        pre_resolved_m3u8: str,
        configuration: DownloadConfigHLS,
    ) -> DownloadReport | bool:
        """
        Threaded HLS segment downloader with optional resume state and stop flag.
        """
        try:
            cleanup_on_stop = configuration.cleanup_on_stop
            keep_segment_dir = configuration.keep_segment_dir
            quality = configuration.quality
            path = configuration.path
            remux = configuration.remux
            start_segment = configuration.start_segment
            segment_state_path = configuration.segment_state_path
            segment_dir = configuration.segment_dir
            return_report = configuration.return_report
            callback = configuration.callback
            callback_remux = configuration.callback_remux
            stop_event = configuration.stop_event
            ios_support = configuration.ios_support
            timeout = timeout
            pre_resolved_m3u8_url = pre_resolved_m3u8

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
            if segment_state_path:
                if os.path.exists(segment_state_path):
                    self.logger.info(f"Found segment state file: {segment_state_path}. Attempting resume.")
                else:
                    self.logger.debug(f"No segment state file found at: {segment_state_path}. Starting fresh.")

            if segment_state_path and os.path.exists(segment_state_path):
                try: # This starts resuming from previous download
                    resume_state = load_segment_state(segment_state_path)
                    resume_mode = True
                except Exception as e: # Shouldn't happen, but if it does, we just do a new download
                    self.logger.warning(f"Failed to load segment state {segment_state_path}: {e}. Starting fresh.")
                    resume_state = None
                    resume_mode = False

            if resume_mode:
                assert resume_state is not None
                segments = resume_state.get("segments") or []  # This fetches the list of segments from the resume state
                if not segments:
                    raise UnknownError("Segment state is invalid or empty.") # Shouldn't happen ;)

                segment_dir = resume_state.get("segment_dir") or segment_dir
                if not segment_dir:
                    raise UnknownError("Segment state is missing segment_dir.")

                created_at = resume_state.get("created_at")
                width = int(resume_state.get("segment_index_width") or get_segment_index_width(len(segments)))
                state_start = int(resume_state.get("start_segment", 0) or 0) # Where we start segments

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
                m3u8_url = resume_state.get("m3u8_url") or ""
                state_quality = resume_state.get("quality", quality)
                self.logger.info(
                    f"Resume state loaded: segments={len(segments)} start_segment={start_segment} "
                    f"segment_dir={segment_dir} segment_index_width={width} created_at={created_at} "
                    f"quality={state_quality} m3u8_url={m3u8_url}"
                )

            else:
                m3u8_master = pre_resolved_m3u8_url
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
                width = get_segment_index_width(len(segments)) if segment_dir else 0
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
            along with the data, we can use that to keep track, since we just change the bool to True for every 
            downloaded segments.
            """

            if segment_dir: # Tries to find existing segments that we already downloaded
                existing_segments = 0
                for i in range(n): # Does that for every segment
                    seg_path = segment_file_path(segment_dir, i, width) # Gets the file path
                    try:
                        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                            # if it exists, we treat it as already downloaded (makes sense)
                            downloaded[i] = True
                            existing_segments += 1
                    except Exception as exc:
                        self.logger.warning(f"Couldn't download segment: {i}, retrying later.  ->: {exc}")
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
            progress_log_step = max(1, n // 20)
            next_progress_log = ((progressed // progress_log_step) + 1) * progress_log_step

            if stop_event is not None and stop_event.is_set():
                cancelled = True
                target_indices = [] # Empty list stops the download :)
                self.logger.warning("Stop event already set; cancelling before scheduling segments.")

            if target_indices:
                workers = max(1, min(max_workers, len(target_indices)))
                parts: List[bytes | None] | None = None
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
                                    _, segment_data, is_success = await self.download_segment(url, timeout, stop_event)
                                    if is_success and segment_data:
                                        return idx, True, segment_data
                                except Exception as exception:
                                    self.logger.error(f"Worker exception for segment {idx}: {exception}")

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
                                seg_path = segment_file_path(segment_dir, i, width)
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
                                await asyncio.to_thread(write_chunks, cast(Any, out_fp), chunks_to_write)

                finally:
                    if out_fp is not None:
                        out_fp.close()

            missing = [i for i, ok in enumerate(downloaded) if not ok] # Missing segments
            missing_urls = [segments[i] for i in missing] # Missing URLs of segments
            self.logger.info(
                "Segment download finished: downloaded=%s/%s missing=%s cancelled=%s",
            downloaded_count, n, len(missing), cancelled)
            if missing:
                sample = missing[:10]
                self.logger.error(
                    "Missing segments detected: count=%s sample=%s", len(missing), sample
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
                    assert  isinstance(segment_state_path, str)
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = build_segment_state(
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
                    write_segment_state(segment_state_path, state)

                if return_report:
                    missing = report.missing
                    self.logger.debug(
                        f"Returning cancelled report: downloaded={report.downloaded} missing={len(missing)}"
                    )
                    return report
                return False

            if missing:
                self.logger.error(
                    f"Download incomplete: {len(missing)} segments missing. Writing state={bool(segment_state_path)}"
                )
                self._safe_remove(tmp_path)
                if segment_state_path:
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = build_segment_state(
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
                    write_segment_state(segment_state_path, state)
                if return_report:
                    self.logger.debug(
                        f"Returning failed report: downloaded={report.downloaded} missing={len(report.missing)}"
                    )
                    return report
                return False

            if segment_dir:
                self.logger.info(
                    f"Assembling {n} segments from {segment_dir} into {tmp_path}"
                )
                def assemble_segments() -> List[int]:
                    with open(tmp_path, "wb") as out_file_path:
                        for idx in range(n):
                            segment_path = segment_file_path(segment_dir, idx, width)
                            if not os.path.exists(segment_path):
                                return [idx]
                            with open(segment_path, "rb") as seg_fp:
                                shutil.copyfileobj(seg_fp, out_file_path, length=1024 * 1024) # type: ignore[arg-type]
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
                        state = build_segment_state(
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
                        write_segment_state(segment_state_path, state)
                    report.status = "failed"
                    report.missing = missing
                    report.missing_urls = [segments[i] for i in missing]
                    if return_report:
                        self.logger.debug(
                            f"Returning failed report after assemble: downloaded={report.downloaded} "
                            f"missing={len(report.missing)}"
                        )
                        return report
                    return False

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
                except Exception as exc: # Shouldn't happen and I also don't know what this does lol
                    self.logger.warning(f"os.replace failed: {exc}, falling back to manual copy.")
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
        except Exception as e:
            self.logger.exception(f"Unhandled exception in download wrapper: {e}")
            return False

    def _convert_ts_to_mp4(self, input_path: str, output_path: str,
                           callback: Callable[[int, int], None] | None = None, ios_support: bool = False) -> None:
        start_ts = time.perf_counter()
        self.logger.info("Remux start: input=%s output=%s", input_path, output_path)

        try:
            input_size = os.path.getsize(input_path)
            self.logger.debug("Remux input size: %s bytes", input_size)
        except Exception as e:
            self.logger.debug("Remux input size unavailable: %s", e)

        try:
            from av import open as av_open  # type: ignore[import-not-found]
            from av.audio.resampler import AudioResampler  # type: ignore[import-not-found]
            import av.audio.frame  # Used for runtime isinstance check
        except (ModuleNotFoundError, ImportError) as e:
            self.logger.error("PyAV import failed for remux: %s", e)
            raise ModuleNotFoundError(
                f"PyAV is required for remuxing. Install with pip install av. Not supported on Termux! {e}")

        self.logger.debug("Opening input for remux: %s", input_path)
        input_ = av_open(input_path)
        fmt_name = (input_.format.name or "").lower()
        self.logger.info("Input format detected: %s", fmt_name or '<unknown>')

        if fmt_name == "mpegts":
            # Fix 1: Suppress the stub mismatch for av.open
            output = av_open(output_path, mode="w", format="mp4",
                             options={"movflags": "faststart"})  # type: ignore[arg-type]

            # --- VIDEO ---
            in_video = input_.streams.video[0]
            out_video = output.add_stream_from_template(template=in_video)
            self.logger.debug(
                "Video stream: codec=%s bit_rate=%s",
                getattr(in_video.codec_context, 'name', None), getattr(in_video.codec_context, 'bit_rate', None)
            )

            # --- AUDIO ---
            in_audio = next((s for s in input_.streams if s.type == "audio"), None)
            out_audio = None
            transcode_audio = False
            resampler = None

            if in_audio:
                # Fix 3: Explicitly narrow out None
                assert in_audio is not None

                # Fix 2: Cast context to AudioCodecContext so IDE knows about sample_rate and layout
                audio_ctx = cast('AudioCodecContext', in_audio.codec_context)

                copy_ok = {"aac"} if ios_support else {"aac", "alac", "mp3"}
                codec_name = (audio_ctx.name or "").lower()
                sample_rate = audio_ctx.sample_rate or 0
                layout_name = audio_ctx.layout.name if getattr(audio_ctx, "layout", None) else "unknown"

                self.logger.debug(
                    "Audio stream: codec=%s sample_rate=%s layout=%s", codec_name, sample_rate, layout_name
                )

                if codec_name in copy_ok:
                    out_audio = output.add_stream_from_template(template=in_audio)
                    self.logger.info("Audio codec MP4-compatible; remuxing without transcoding.")
                else:
                    transcode_audio = True
                    sample_rate = audio_ctx.sample_rate or 48000
                    layout = audio_ctx.layout.name if getattr(audio_ctx, "layout", None) else "stereo"

                    out_audio = output.add_stream("aac", rate=sample_rate)
                    self.logger.info("Transcoding audio to AAC: sample_rate=%s layout=%s"), sample_rate, layout

                    try:
                        out_audio.layout = layout
                    except Exception as exc:
                        self.logger.warning("Exception in getting audio layout (doesn't matter): %s", exc)
                        pass

                    resampler = AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            else:
                self.logger.info("No audio stream detected; remuxing video only.")

            # --- DEMUX ---
            demux_streams = [in_video] + ([in_audio] if in_audio else [])
            packets = input_.demux(demux_streams)

            try:
                total = os.path.getsize(input_path)
            except Exception as exc:
                self.logger.warning("Exception while getting path size for demuxing progress??? %s", exc)
                total = 100

            self.logger.info("Demuxing packets: total_bytes=%s", total)
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
                    packet.stream = out_video
                    output.mux(packet)

                elif in_audio and packet.stream == in_audio:
                    if not transcode_audio:
                        packet.stream = out_audio
                        output.mux(packet)
                    else:
                        assert out_audio is not None
                        for frame in packet.decode():
                            # Fix 4: Ensure the frame is recognized as an AudioFrame
                            if not isinstance(frame, av.audio.frame.AudioFrame):
                                continue

                            frames = resampler.resample(frame) if resampler else [frame]
                            for f in frames:
                                for enc_pkt in out_audio.encode(f):
                                    output.mux(enc_pkt)

                if callback:
                    callback(current_progress, total)
                if progress_step and current_progress >= next_progress_log:
                    self.logger.debug("Remux progress: bytes=%s/%s", current_progress, total)
                    next_progress_log += progress_step

            if transcode_audio and out_audio:
                self.logger.debug("Flushing AAC encoder.")
                for enc_pkt in out_audio.encode(None):
                    output.mux(enc_pkt)

            input_.close()
            output.close()
            elapsed = time.perf_counter() - start_ts

            try:
                out_size = os.path.getsize(output_path)
                self.logger.info("Remux complete: output=%s size=%s bytes elapsed=%s.2f", output_path, out_size,
                                 elapsed)
            except Exception as e:
                self.logger.info("Remux complete: output=%s elapsed=%s.2fs (size unavailable: %s)", output_path,
                                 elapsed, e)

        else:
            self.logger.info("Stream seems to be already in MP4! Skipping remux...")
            os.rename(input_path, output_path)
            elapsed = time.perf_counter() - start_ts
            self.logger.info("Remux skipped; file moved. elapsed=%s.2f", elapsed)

    async def legacy_download(self, url: str, configuration: DownloadConfigRAW) -> bool:
        """
        Download a file using streaming with stall tolerance and resume.
        Supports fast concurrent range downloading if the server supports it and allow_multipart is True.
        Assumes self.session is an AsyncSession.
        """
        path = configuration.path
        max_retries = configuration.max_retries
        read_timeout = configuration.read_timeout
        stop_event = configuration.stop_event
        allow_multipart = configuration.allow_multipart
        callback = configuration.callback
        chunk_size = configuration.chunk_size
        max_workers = configuration.max_workers

        self.logger.info(
"""Legacy download start: url=%s path=%s
max_retries=%s read_timeout=%s
stop_event_set=%s
allow_multipart=%s""", url, path, max_retries, read_timeout, bool(stop_event and stop_event.is_set()),
        allow_multipart)

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
                    head_resp_stream = await session.request("GET", url, timeout=timeout, allow_redirects=True,
                                                             stream=True, headers=no_compress)
                    file_size = int(head_resp_stream.headers.get("Content-Length", 0))
                    accept_ranges = head_resp_stream.headers.get("Accept-Ranges", "")
                else:
                    file_size = int(head_resp.headers.get("Content-Length", 0))
                    accept_ranges = head_resp.headers.get("Accept-Ranges", "")
            except Exception as e:
                self.logger.warning("Failed to fetch HEAD info for concurrent check: %s.", e)

        # 2. Execute Fast Multipart Download if supported and allowed
        if allow_multipart and file_size > 0 and accept_ranges == "bytes":
            self.logger.info("Server supports Range requests. Starting fast multipart download"
                             "or %s bytes.", file_size)

            # Pre-allocate file
            def allocate_file() -> None:
                if not os.path.exists(path):
                    with open(path, "wb") as file_alloc:
                        file_alloc.truncate(file_size)
                elif os.path.getsize(path) != file_size:
                    # File exists but size mismatch, truncate to correct size
                    with open(path, "r+b") as file_alloc_size:
                        file_alloc_size.truncate(file_size)
            await asyncio.to_thread(allocate_file)

            # We will use an array to track progress of chunks
            # A chunk map: {chunk_index: bytes_downloaded}
            chunk_progress = {}
            total_downloaded = [0]  # List to allow modification in inner func
            # Determine chunk sizes based on file size, but keep reasonable bounds
            # For massive files, don't create 10,000 workers.
            target_chunk_size = max(chunk_size, min(10 * 1024 * 1024, file_size // 10)) # Between 1MB and 10MB

            semaphore = asyncio.Semaphore(max_workers)

            async def download_chunk(start_chunk: int, end_chunk: int, chunk_idx_now: int) -> bool:
                nonlocal total_downloaded
                headers_chunk = {"Range": f"bytes={start_chunk}-{end_chunk}", "Accept-Encoding": "identity"}
                chunk_progress[chunk_idx_now] = 0

                for attempt_chunk in range(max_retries + 1):
                    if stop_event is not None and stop_event.is_set():
                        return False

                    try:
                        async with semaphore:
                            resp = await cast(Any, session).request(
                                "GET", url, headers=headers_chunk, timeout=timeout, allow_redirects=True, stream=True
                            )
                            resp.raise_for_status()

                            # Open file once for this chunk download attempt
                            file = await asyncio.to_thread(lambda: open(path, "rb+"))
                            try:
                                await asyncio.to_thread(file.seek, start_chunk + chunk_progress[chunk_idx_now])
                                async for data in resp.aiter_content():
                                    if stop_event is not None and stop_event.is_set():
                                        return False

                                    await asyncio.to_thread(cast(Any, file).write, data)

                                    data_len = len(data)
                                    chunk_progress[chunk_idx_now] += data_len
                                    total_downloaded[0] += data_len

                                    if callback:
                                        callback(total_downloaded[0], file_size)
                                    elif progress_bar:
                                        progress_bar.text_progress_bar(downloaded=total_downloaded[0], total=file_size)
                            finally:
                                await asyncio.to_thread(file.close)

                            return True # Chunk success

                    except Exception as exc:
                        if attempt_chunk < max_retries:
                            self.logger.warning("Chunk %s failed (attempt %s/%s): %s",
                                                chunk_idx_now, attempt_chunk + 1, max_retries, exc)
                            # Reset progress for this chunk before retry
                            total_downloaded[0] -= chunk_progress[chunk_idx_now]
                            chunk_progress[chunk_idx_now] = 0
                            await asyncio.sleep(1 * attempt_chunk)
                        else:
                            self.logger.error("Chunk %s permanently failed: %s", chunk_idx_now, exc)
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
                # We set it to None instead of del to avoid analyzer confusion about potential unassigned reference
                progress_bar = None

            if stop_event is not None and stop_event.is_set():
                raise DownloadCancelled("Download cancelled.")

            if not all(results):
                raise NetworkRequestError("One or more chunks failed to download completely.")

            self.logger.info("Fast multipart download complete: path=%s", path)
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
                        await asyncio.to_thread(f.write, chunk)
                        downloaded_so_far += len(chunk)

                        if callback:
                            callback(downloaded_so_far, total)
                        elif progress_bar:
                            progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=total)
                finally:
                    await asyncio.to_thread(f.close)

                if progress_bar:
                    progress_bar = None
                self.logger.info("Legacy download complete: bytes=%s path=%s", downloaded_so_far, path)
                return True
            except RequestsError as e:
                err_str = str(e).lower()
                if "timeout" in err_str or "read" in err_str:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    backoff = min(2 ** attempt, 30)
                    self.logger.warning("Read timeout; retrying %s/%s in %s", attempt, max_retries,  backoff)
                    if stop_event is not None and stop_event.wait(backoff):
                        raise DownloadCancelled("Download cancelled.")
                    else:
                        await asyncio.sleep(backoff)
                    continue
                else:
                    raise NetworkRequestError(f"Stream for: {url} was closed or failed: {e}")
            except DownloadCancelled:
                raise
            except Exception:
                error = traceback.format_exc()
                raise NetworkRequestError(f"Unknown error for: {url} -->: {error}")

        return False