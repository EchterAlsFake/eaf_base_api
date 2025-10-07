import re
import os
import sys
import time
import m3u8
import httpx
import random
import logging
import traceback
import threading

from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from modules.errors import *
    from modules.config import config
    from modules.progress_bars import Callback

except (ModuleNotFoundError, ImportError):
    from .modules.errors import *
    from .modules.config import config
    from .modules.progress_bars import Callback

user_agents = [
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.3",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.3",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0."
        ]

loggers = {}
HEIGHT_FROM_URI = re.compile(r'(?<!\d)(\d{3,4})[pP](?!\d)')  # e.g., 1080p, 720P

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

_CF_BODY_MARKERS = (
    "cf-browser-verification",
    "cf_chl_",  # legacy CF challenge tokens
    "challenges.cloudflare.com",  # Turnstile/challenge host
    "cf-turnstile",  # Turnstile widget
    "ddos protection by",
    "just a moment...",
    "please enable javascript and cookies",
    "attention required!",
)
# Other common WAF/CDN markers
_OTHER_BODY_MARKERS = (
    "akamai bot manager",
    "bm_sz=", "abck=", "ak_bmsc", "ak_bm",                # Akamai cookies
    "request unsuccessful. incapsula incident id",        # Imperva/Incapsula
)

def _is_text_html(resp) -> bool:
    ct = resp.headers.get("content-type", "").lower()
    return "text/html" in ct or ct == ""  # some sites omit CT but send HTML

def _header_has_any(hdr_val: str, *needles: str) -> bool:
    s = (hdr_val or "").lower()
    return any(n in s for n in needles)

def _body_has_any(body: str, markers: tuple) -> bool:
    b = (body or "").lower()
    return any(m in b for m in markers)

def _looks_like_challenge(resp) -> str | None:
    """
    Return a human-readable reason if the response looks like a bot/WAF challenge.
    Return None if it looks like a normal page.
    """
    status = resp.status_code
    server = resp.headers.get("server", "")
    set_cookie = resp.headers.get("set-cookie", "")

    # Collect cheap, header-based indicators first
    has_cf_cookie = _header_has_any(set_cookie, "__cf_bm", "cf_clearance", "__cf_chl")
    served_by_cf = _header_has_any(server, "cloudflare")
    served_by_akamai = _header_has_any(server, "akamai")
    served_by_incapsula = _header_has_any(server, "incapsula")

    # History-based indicators (redirect to /cdn-cgi/ etc.)
    history_urls = []
    try:
        for h in getattr(resp, "history", []) or []:
            history_urls.append(str(getattr(h, "url", "")))
    except Exception:
        pass
    redirected_via_cdn_cgi = any("/cdn-cgi/" in u for u in history_urls)

    # Read a *small* slice of the body only if it’s HTML-like
    body_snippet = ""
    if _is_text_html(resp):
        try:
            # Read up to 20KB to capture challenge markup without huge memory hit
            text = resp.text
            body_snippet = text[:20000]
        except Exception:
            body_snippet = ""

    has_cf_body_markers = _body_has_any(body_snippet, _CF_BODY_MARKERS)
    has_other_waf_markers = _body_has_any(body_snippet, _OTHER_BODY_MARKERS)

    # Heuristics:
    # 1) Hard failure statuses usually mean a challenge/waf
    if status in (403, 429, 503):
        if served_by_cf or has_cf_cookie or redirected_via_cdn_cgi or has_cf_body_markers or has_other_waf_markers:
            return (f"HTTP {status} with WAF indicators "
                    f"(server={server!r}, cf_cookie={has_cf_cookie}, "
                    f"cdn_cgi_redirect={redirected_via_cdn_cgi}, body_markers={has_cf_body_markers or has_other_waf_markers})")

    # 2) Status 200 but body is a *challenge/interstitial*
    #    Require strong evidence in the body or cookies, not just 'Server: cloudflare'.
    if status == 200:
        if has_cf_cookie or redirected_via_cdn_cgi or has_cf_body_markers or has_other_waf_markers:
            return (f"200 OK but challenge content detected "
                    f"(server={server!r}, cf_cookie={has_cf_cookie}, "
                    f"cdn_cgi_redirect={redirected_via_cdn_cgi}, body_markers={has_cf_body_markers or has_other_waf_markers})")

    # 3) Generic: if served by a known WAF/CDN and body is *obviously* an interstitial
    if served_by_cf and has_cf_body_markers:
        return "Cloudflare challenge markers present in body"
    if served_by_akamai and has_other_waf_markers:
        return "Akamai Bot Manager markers present in body"
    if served_by_incapsula and has_other_waf_markers:
        return "Imperva/Incapsula challenge markers present in body"

    return None

def is_android():
    """Detects if the script is running on an Android device."""
    return "ANDROID_ROOT" in os.environ and "ANDROID_DATA" in os.environ

def get_log_file_path(filename="app.log"):
    """Returns a valid log file path that works on Android and other OS."""
    if is_android():
        return os.path.join(os.environ["HOME"], filename)  # Internal app storage
    return filename  # Default for Linux, Windows, Mac


# Define ANSI color codes for different log levels
LOG_COLORS = {
    'DEBUG': "\033[37m",   # White
    'INFO': "\033[34m",    # Blue
    'WARNING': "\033[33m", # Yellow
    'ERROR': "\033[31m",   # Red
    'CRITICAL': "\033[41m"  # Red background
}
RESET_COLOR = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        color = LOG_COLORS.get(record.levelname, RESET_COLOR)
        return f"{color}{log_message}{RESET_COLOR}"


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

class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(self, config=config, auto_init: bool = True, headers: dict = None):
        self.last_request_time = time.time()
        self.total_requests = 0 # Tracks how many requests have been made
        self.session = None
        self.kill_switch = False
        self.config = config
        self.cache = Cache(self.config)
        self.logger = setup_logger("BASE API - [BaseCore]", log_file=False, level=logging.ERROR)
        if auto_init:
            self.initialize_session(headers)

    def enable_logging(self, log_file=None, level=logging.DEBUG, log_ip=None, log_port=None):
        """
        Enables logging dynamically for this module.
        """
        self.logger = setup_logger(name="BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)
        self.cache.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)

    def update_cookies(self):
        """Updates cookies dynamically"""
        self.session.cookies.update(self.config.cookies)

    def enable_kill_switch(self):
        """This is a function that will check and verify if your proxy is working before every request.
        As soon as there's a mismatch in the actual response IP and the proxy IP, the program will exit."""
        self.kill_switch = True

    def check_kill_switch(self):
        proxy_ip = self.config.proxy
        pattern = re.compile(
            r'^(?P<scheme>http|socks5)://'
            r'(?:\w+:\w+@)?'  # optional user:pass@
            r'(?P<host>[a-zA-Z0-9.-]+)'
            r':(?P<port>\d{1,5})$'
        )

        match = pattern.match(proxy_ip
    )
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


    def initialize_session(self, headers: dict = None, cookies: dict = None): # Disable SSL verification only if you really need it....
        self.session = httpx.Client(
            proxy=self.config.proxy,
            timeout=self.config.timeout,
            follow_redirects=True,
            verify=self.config.verify_ssl
        )

        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"})

        if headers:
            self.session.headers.update(headers)

        if cookies:
            self.session.cookies.update(cookies)

    def update_headers(self, headers: dict):
        self.session.headers.update(headers)

    def enforce_delay(self):
        """Enforces the specified delay in consts.REQUEST_DELAY"""
        delay = self.config.request_delay
        if delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            self.logger.debug(f"Time since last request: {time_since_last_request:.2f} seconds.")
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                self.logger.debug(f"Enforcing delay of {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
        self.last_request_time = time.time()

    def fetch(
            self,
            url: str,
            get_bytes: bool = False,
            timeout: int = None,
            get_response: bool = False,
            save_cache: bool = True,
            cookies: dict = None,
            allow_redirects: bool = True,
            bypass_kill_switch: bool = False, # prevents infinite loop
            data: dict = None,
            method: str = "GET",
            headers: dict = None,
            json: dict = None,
    ) -> Union[bytes, str, httpx.Response, None]:
        """
        Fetches content in UTF-8 Text, Bytes, or as a stream using multiple request attempts,
        support for proxies and custom timeout.
        """
        # Check cache first
        last_error = None

        if timeout is None:
            timeout = self.config.timeout # pls don't ask thanks

        content = self.cache.handle_cache(url)
        if content:
            self.logger.info(f"Fetched content for: {url} from cache!")
            return content

        for attempt in range(1, self.config.max_retries + 1): # +1 because I feel like it
            if attempt != 1:
                time.sleep(1.5 * attempt) # Sleeping for 1.5 seconds to minimize site overload when doing a lot of requests

            try:
                self.enforce_delay()

                if self.kill_switch and not bypass_kill_switch:
                    self.check_kill_switch()

                # Perform the request with stream handling
                if headers is None:
                    headers = self.session.headers.copy()

                response = self.session.request(method=method, url=url, timeout=timeout, cookies=cookies,
                                     follow_redirects=allow_redirects, data=data, headers=headers, json=json)
                self.total_requests += 1

                # NEW: bot/challenge detection
                reason = _looks_like_challenge(response)
                if reason:
                    # Log a concise diagnostic with a small body preview
                    body_preview = ""
                    try:
                        if _is_text_html(response):
                            body_preview = response.text[:400]
                    except Exception:
                        pass
                    self.logger.error(
                        "Bot protection detected for %s: %s | Body[:400]=%r",
                        url, reason, body_preview
                    )
                    if config.raise_bot_protection:
                        raise BotProtectionDetected(f"{reason} at {url}")

                # Log and handle non-200 status codes
                if response.status_code != 200:
                    self.logger.warning(
                        f"Attempt {attempt}: Unexpected status code {response.status_code} for URL: {url}")

                    if response.status_code == 404:
                        self.logger.error(f"URL: {url} Resource not found (404). This may indicate the content is unavailable.")
                        return response  # Return None for unavailable resources

                    if response.status_code == 403:
                        time.sleep(2) # Somehow fixes 403 on missav idk man
                        self.session.headers.update({
                            "User-Agent": random.choice(user_agents)
                        })

                        self.logger.warning(f"Switched User agent to: {self.session.headers['User-Agent']}")


                    elif response.status_code == 403 and attempt >= 2:
                        self.logger.error(f"The website rejected access after {attempt} tries. Aborting!")
                        return None # Return None for forbidden resources

                    elif response.status_code == 410:
                        raise ResourceGone(f"Resource: {url} is Gone.")

                    self.logger.info(f"Retrying ({attempt}/{self.config.max_retries}) for URL: {url}")
                    continue  # Retry for other non-200 status codes

                self.logger.debug(f"Attempt {attempt}: Successfully fetched URL: {url}")

                # Return response if requested
                if get_response:
                    return response

                # Process and return content
                # Collecting all chunks before processing because some sites cause issues with real-time decoding
                if self.config.max_bandwidth_mb is not None and self.config.max_bandwidth_mb >= 0.2:
                    raw_content = bytearray()
                    chunk_size = 64 * 1024 # 64 KB
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
                    try:
                        content = raw_content.decode("utf-8")

                    except UnicodeDecodeError:
                        self.logger.warning(f"Content could not be decoded as utf-8 ({url}), decoding in 'latin1' instead!")
                        content = raw_content.decode("latin1") # Fallback, hope that works somehow idk

                    if save_cache and not get_bytes:
                        self.logger.debug(f"Saving content of {url} to local cache.")
                        self.cache.save_cache(url, content)

                return content

            except httpx.CloseError:
                self.logger.error(f"Attempt {attempt}: The connection has been unexpectedly closed by: {url}. Retrying...")
                continue

            except (httpx.RequestError, httpx.ConnectError) as e:
                self.logger.error(f"Attempt {attempt}: Request error for URL {url}: {e}")

                if "CERTIFICATE_VERIFY_FAILED" in str(e):
                    raise ProxySSLError("Proxy has an invalid SSL certificate, to get around this error, set 'verify = False' in config")

                self.logger.info(f"Retrying ({attempt}/{self.config.max_retries}) for URL: {url}")
                continue # Continuing even failed requests, because they may succeed on another try (maybe idk)

            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                self.logger.error(f"Attempt {attempt}: Timeout error for URL {url}: {e}. Please increase your timeout in "
                                  f"the configuration options to prevent that, or get a stable internet connection ;)")
                self.logger.info(f"Retrying ({attempt}/{self.config.max_retries}) for URL: {url}")
                continue

            except httpx.CookieConflict as e:
                self.logger.error(f"Some cookies are conflicting. Connection aborted! Please report this error immediately -->: {e}")
                sys.exit(1)

            except httpx.ProxyError as e:
                self.logger.error(f"Proxy Error, please switch to another Proxy, it seems to be bad -->: {e}")
                raise KillSwitch("Proxy error when trying a request, Aborting!")

            except Exception as e:
                last_error = e
                self.logger.error(f"Attempt {attempt}: Unexpected error for URL {url}: {traceback.format_exc()}")
                raise UnknownError(f"Unexpected error for URL {url}: {traceback.format_exc()}")

        self.logger.error(f"Failed to fetch URL {url} after {self.config.max_retries} attempts.")
        if last_error:
            pass
        return None  # Return None if all attempts fail

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

    def get_segments(self, m3u8_url_master: str, quality: Union[str, int]) -> list:
        m3u8_url = self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        self.logger.debug(f"Trying to fetch segment from m3u8 -> {m3u8_url}")
        content = self.fetch(url=m3u8_url)

        segments = []
        m3u8_processed = m3u8.loads(content)

        if m3u8_processed.is_variant:
            self.logger.warning("Media playlist expected; got variant. Resolving to first sub-playlist...")
            new_m3u8 = urljoin(m3u8_url, m3u8_processed.playlists[0].uri)
            self.logger.info(f"Resolved to new URL: {new_m3u8}")
            content = self.fetch(url=new_m3u8)
            m3u8_processed = m3u8.loads(content)

        if getattr(m3u8_processed, "segment_map", None):
            init_uri = m3u8_processed.segment_map[0].uri
            init_url = urljoin(m3u8_url, init_uri)
            segments.append(init_url)
            self.logger.debug(f"Found init segment: {init_url}")

        segments.extend(urljoin(m3u8_url, seg.uri) for seg in m3u8_processed.segments)
        self.logger.debug(f"Fetched {len(segments)} segments from m3u8 URL (including init if present)")
        return segments

    def download_segment(self, url: str, timeout: int) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment, retrying on failure.
        Returns a tuple of the URL, content (empty if failed after retries), and a success flag.
        """
        content = self.fetch(url, timeout=timeout, get_bytes=True)
        return url, content, True  # Success

    def download(self, video, quality: str, downloader: str, path: str, callback=None, remux: bool = False,
                 callback_remux=None) -> None:
        """
        :param video:
        :param callback:
        :param downloader:
        :param quality:
        :param path:
        :param remux:
        :param callback_remux:
        :return:
        """

        if callback is None:
            callback = Callback.text_progress_bar

        if downloader == "default":
            self.default(video=video, quality=quality, path=path, callback=callback, remux=remux, callback_remux=callback_remux)

        elif downloader == "threaded":
            threaded_download = self.threaded(max_workers=20, timeout=10)
            threaded_download(self, video=video, quality=quality, path=path, callback=callback, remux=remux,
                              callback_remux=callback_remux)

        elif downloader == "FFMPEG":
            self.FFMPEG(video=video, quality=quality, path=path, callback=callback)

    def threaded(self, max_workers: int = 20, timeout: int = 10):
        def wrapper(self, video, quality: str, callback, path: str, remux: bool = True, callback_remux=None):
            segments = self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)
            length = len(segments)
            completed = 0
            ts_buffers = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {
                    executor.submit(self.download_segment, url, timeout): url for url in segments
                }

                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        _, data, success = future.result()
                        completed += 1
                        if success:
                            ts_buffers.append((segments.index(url), data))
                        callback(completed, length)
                    except Exception as e:
                        self.logger.error(f"Error processing segment {url}: {e}")
                        raise SegmentError(f"Error processing segment {url}: {e}")

            ts_buffers.sort(key=lambda x: x[0])
            ts_combined = b''.join(data for _, data in ts_buffers)

            if remux:
                tmp_path = f"{path}.tmp"
                with open(tmp_path, 'wb') as file:
                    file.write(ts_combined)
                self._convert_ts_to_mp4(tmp_path, path, callback=callback_remux)
                os.remove(tmp_path)
            else:
                with open(path, 'wb') as file:
                    file.write(ts_combined)

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

    def FFMPEG(self, video, quality: str, callback, path: str) -> bool:
        try:
            from ffmpeg_progress_yield import FfmpegProgress

        except (ModuleNotFoundError, ImportError):
            raise ModuleNotFoundError("To use FFmpeg progress, you need to install ffmpeg-progress-yield. "
                                      "You can do so with: `pip install ffmpeg-progress-yield`")
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

    @classmethod
    def return_path(cls, video, args) -> str:
        path = args.output
        if args.use_title:
            if not str(path).endswith(os.sep):
                path += os.sep
            path += video.title + ".mp4"
        else:
            path = args.output

        return path

    def truncate(self, name: str, max_bytes: int = 245) -> str: # only 245, because we need to append .mp4
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
