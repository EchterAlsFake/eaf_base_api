import re
import os
import sys
import time
import m3u8
import httpx
import logging
import traceback
import threading

from typing import Union
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

loggers = {}

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
    def __init__(self, config=config):
        self.last_request_time = time.time()
        self.total_requests = 0 # Tracks how many requests have been made
        self.session = None
        self.kill_switch = False
        self.config = config
        self.cache = Cache(self.config)
        self.initialize_session()
        self.logger = setup_logger("BASE API - [BaseCore]", log_file=False, level=logging.ERROR)

    def enable_logging(self, log_file=None, level=logging.DEBUG, log_ip=None, log_port=None):
        """
        Enables logging dynamically for this module.
        """
        self.logger = setup_logger(name="BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)
        self.cache.logger = setup_logger(name="BASE API - [Cache]", log_file=log_file, level=level, http_ip=log_ip, http_port=log_port)

    def update_user_agent(self):
        """Updates the User-Agent"""
        self.config.rotate_user_agent()
        self.session.headers.update({"User-Agent": self.config.headers["User-Agent"]})

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
            r'(?:(?:\w+:\w+)@)?'  # optional user:pass@
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


    def initialize_session(self, verify=True): # Disable SSL verification only if you really need it....
        self.session = httpx.Client(
            proxy=self.config.proxy,
            headers=self.config.headers,
            timeout=self.config.timeout,
            follow_redirects=True,
            verify=verify
        )

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
    ) -> Union[bytes, str, httpx.Response, None]:
        """
        Fetches content in UTF-8 Text, Bytes, or as a stream using multiple request attempts,
        support for proxies and custom timeout.
        """
        # Check cache first

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
                # Update user agent periodically
                if self.total_requests % 30 == 0: # Changing all 30 attempts
                    self.update_user_agent()

                self.enforce_delay()

                if self.kill_switch and not bypass_kill_switch:
                    self.check_kill_switch()

                # Perform the request with stream handling
                response = self.session.request(method=method, url=url, timeout=timeout, cookies=cookies,
                                     follow_redirects=allow_redirects, data=data)
                self.total_requests += 1

                # Log and handle non-200 status codes
                if response.status_code != 200:
                    self.logger.warning(
                        f"Attempt {attempt}: Unexpected status code {response.status_code} for URL: {url}")

                    if response.status_code == 404:
                        self.logger.error(f"URL: {url} Resource not found (404). This may indicate the content is unavailable.")
                        return response  # Return None for unavailable resources

                    elif response.status_code == 403 and attempt >= 2:
                        self.logger.error(f"The website rejected access after {attempt} tries. Aborting!")
                        return None # Return None for forbidden resources

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

            except httpx.RequestError as e:
                self.logger.error(f"Attempt {attempt}: Request error for URL {url}: {e}")
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

            except Exception:
                self.logger.error(f"Attempt {attempt}: Unexpected error for URL {url}: {traceback.format_exc()}")
                raise UnknownError(f"Unexpected error for URL {url}: {traceback.format_exc()}")

        self.logger.error(f"Failed to fetch URL {url} after {self.config.max_retries} attempts.")
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
    def get_m3u8_by_quality(self, m3u8_url: str, quality: str) -> str:
        """Fetches the m3u8 URL for a given quality by extracting all possible sub-m3u8 URLs from the primary
        m3u8 playlist"""
        if "#" in m3u8_url:
            playlist = m3u8.loads(m3u8_url)
            self.logger.debug("Resolved custom m3u8")

        else:
            playlist_content = self.fetch(url=m3u8_url)
            playlist = m3u8.loads(playlist_content)
            self.logger.debug(f"Resolved m3u8 playlist: {m3u8_url}")

        if not playlist.is_variant:
            raise ValueError("Provided URL is not a master playlist.")

        # Extract available qualities and URLs
        qualities = {
            (variant.stream_info.resolution or "unknown"): variant.uri
            for variant in playlist.playlists
        } # Fetches the available qualities

        # Sort qualities by resolution (width x height)
        sorted_qualities = sorted(
            qualities.items(),
            key=lambda x: (x[0][0] * x[0][1]) if x[0] != "unknown" else 0
        ) # Sorts qualities to support different HLS naming schemes

        # Select based on quality preference
        if quality == "best":
            url_tmp = sorted_qualities[-1][1]  # Highest resolution URL

        elif quality == "worst":
            url_tmp = sorted_qualities[0][1]  # Lowest resolution URL

        elif quality == "half":
            url_tmp = sorted_qualities[len(sorted_qualities) // 2][1]  # Mid-quality URL

        else:
            raise ValueError("Invalid quality provided.")

        full_url = urljoin(m3u8_url, url_tmp) # Merges the primary with the new URL to get the full URL path
        return full_url

    @lru_cache(maxsize=250)
    def get_segments(self, m3u8_url_master: str, quality: str) -> list:
        """Gets all video segments from a given quality and the primary m3u8 URL"""
        m3u8_url = self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        self.logger.debug(f"Trying to fetch segment from m3u8 ->: {m3u8_url}")
        content = self.fetch(url=m3u8_url)

        m3u8_processed = m3u8.loads(content)

        if m3u8_processed.is_variant:
            self.logger.warning("The M3U8 Playlist is not a media playlist. Trying to resolve to the first m3u8...")
            new_m3u8 = urljoin(m3u8_url, m3u8_processed.playlists[0].uri)
            self.logger.info(f"Resolved to new URL: {new_m3u8}")
            content = self.fetch(url=new_m3u8)
            m3u8_processed = m3u8.loads(content)

        segments = [urljoin(m3u8_url, seg.uri) for seg in m3u8_processed.segments]
        self.logger.debug(f"Fetched {len(segments)} segments from m3u8 URL")
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
            import av
        except (ModuleNotFoundError, ImportError):
            raise ModuleNotFoundError("PyAV is required for remuxing. Install with `pip install av`. Not supported on Termux!")

        input_ = av.open(input_path)
        output = av.open(output_path, "w")

        in_stream = input_.streams.video[0]
        out_stream = output.add_stream_from_template(template=in_stream)

        packets = list(input_.demux(in_stream))
        total = len(packets)

        for idx, packet in enumerate(packets):
            if packet.dts is None:
                continue
            packet.stream = out_stream
            output.mux(packet)
            if callback:
                callback(idx + 1, total)

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

    def legacy_download(self, path: str, url: str, callback=None) -> bool:
        """
        Download a file using streaming, with support for progress updates.
        """
        try:
            with self.session.stream("GET", url, timeout=30) as response:  # Use a reasonable timeout
                response.raise_for_status()
                file_size = int(response.headers.get('content-length', 0))

                if callback is None:
                    progress_bar = Callback()

                downloaded_so_far = 0

                # Open file for writing
                with open(path, 'wb') as file:
                    for chunk in response.iter_bytes(chunk_size=1024):
                        if not chunk:
                            break  # End of stream

                        file.write(chunk)
                        downloaded_so_far += len(chunk)

                        # Update progress
                        if callback:
                            callback(downloaded_so_far, file_size)
                        else:
                            progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=file_size)

                if not callback:
                    del progress_bar

                return True

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

    @classmethod
    def truncate(cls, string_lol) -> str:
        return string_lol[:200]

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
