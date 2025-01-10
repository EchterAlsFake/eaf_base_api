import os
import time
import m3u8
import httpx
import random
import string
import logging
import traceback
import threading

from typing import Union
from functools import lru_cache
from urllib.parse import urljoin
from ffmpeg_progress_yield import FfmpegProgress
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from modules import consts
    from modules.progress_bars import Callback

except (ModuleNotFoundError, ImportError):
    from .modules import consts
    from .modules.progress_bars import Callback

logging.basicConfig(format='%(name)s %(levelname)s %(asctime)s %(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger("BASE API")
logger.setLevel(logging.DEBUG)

def disable_logging():
    logger.setLevel(logging.CRITICAL)


class Cache:
    """
    Caches content from network requests
    """

    def __init__(self):
        self.cache_dictionary = {}
        self.lock = threading.Lock()

    def handle_cache(self, url):
        if url is None:
            return

        with self.lock:
            content = self.cache_dictionary.get(url, None)
            return content

    def save_cache(self, url, content):
        with self.lock:
            if len(self.cache_dictionary.keys()) >= consts.MAX_CACHE_ITEMS:
                first_key = next(iter(self.cache_dictionary))
                # Delete the first item
                del self.cache_dictionary[first_key]
                logger.info(f"Deleting: {first_key} from cache, due to caching limits...")

            self.cache_dictionary[url] = content

cache = Cache()


class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(self):
        self.last_request_time = time.time()
        self.total_requests = 0 # Tracks how many requests have been made
        self.session = None
        self.initialize_session()


    def update_user_agent(self):
        """Updates the User-Agent"""
        self.session.headers.update({"User-Agent": random.choice(consts.USER_AGENTS)})

    def initialize_session(self):
        verify = True

        if consts.PROXY is not None:
            verify = False

        self.session = httpx.Client(proxy=consts.PROXY,
                              headers=consts.HEADERS,
                              verify=verify,
                              timeout=consts.TIMEOUT
                              )

    def enforce_delay(self):
        """Enforces the specified delay in consts.REQUEST_DELAY"""
        delay = consts.REQUEST_DELAY
        if delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            logger.debug(f"Time since last request: {time_since_last_request:.2f} seconds.")
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                logger.debug(f"Enforcing delay of {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

    def fetch(
            self,
            url: str,
            get_bytes: bool = False,
            timeout: int = consts.TIMEOUT,
            get_response: bool = False,
            save_cache: bool = True,
            cookies: dict = None,
            allow_redirects: bool = True,
    ) -> Union[bytes, str, httpx.Response, None]:
        """
        Fetches content in UTF-8 Text, Bytes, or as a stream using multiple request attempts,
        support for proxies and custom timeout.
        """
        # Check cache first
        content = cache.handle_cache(url)
        if content is not None:
            logger.info(f"Fetched content for: {url} from cache!")
            return content

        for attempt in range(1, consts.MAX_RETRIES + 1):
            try:
                # Update user agent periodically
                if self.total_requests % 3 == 0:
                    self.update_user_agent()

                self.enforce_delay()

                # Perform the request with stream handling
                with self.session.stream("GET", url, timeout=timeout, cookies=cookies,
                                         follow_redirects=allow_redirects) as response:
                    self.total_requests += 1

                    # Log and handle non-200 status codes
                    if response.status_code != 200:
                        logger.warning(
                            f"Attempt {attempt}: Unexpected status code {response.status_code} for URL: {url}")

                        if response.status_code == 404:
                            logger.error("Resource not found (404). This may indicate the content is unavailable.")
                            return None  # Return None for unavailable resources

                        continue  # Retry for other non-200 status codes

                    logger.debug(f"Attempt {attempt}: Successfully fetched URL: {url}")

                    # Return response if requested
                    if get_response:
                        return response

                    # Process and return content
                    if get_bytes:
                        content = b"".join([chunk for chunk in response.iter_bytes()])
                    else:
                        content = "".join([chunk.decode("utf-8") for chunk in response.iter_bytes()])
                        if save_cache:
                            logger.debug(f"Saving content of {url} to local cache.")
                            cache.save_cache(url, content)

                    return content

            except httpx.RequestError as e:
                logger.error(f"Attempt {attempt}: Request error for URL {url}: {e}")

            except Exception:
                logger.error(f"Attempt {attempt}: Unexpected error for URL {url}: {traceback.format_exc()}")

            logger.info(f"Retrying ({attempt}/{consts.MAX_RETRIES}) for URL: {url}")

        logger.error(f"Failed to fetch URL {url} after {consts.MAX_RETRIES} attempts.")
        return None  # Return None if all attempts fail

    @classmethod
    def strip_title(cls, title: str) -> str:
        """
        :param title:
        :return: str: strips out non UTF-8 chars of the title
        """
        illegal_chars = '<>:"/\\|?*'
        cleaned_title = ''.join([char for char in title if char in string.printable and char not in illegal_chars])
        return cleaned_title

    @lru_cache(maxsize=250)
    def get_m3u8_by_quality(self, m3u8_url: str, quality: str) -> str:
        """Fetches the m3u8 URL for a given quality by extracting all possible sub-m3u8 URLs from the primary
        m3u8 playlist"""
        playlist_content = self.fetch(url=m3u8_url)
        playlist = m3u8.loads(playlist_content)
        logger.debug(f"Resolved m3u8 playlist: {m3u8_url}")

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
        logger.debug(f"Trying to fetch segment from m3u8 ->: {m3u8_url}")
        content = self.fetch(url=m3u8_url)
        segments_ = m3u8.loads(content).segments
        segments = []

        for segment in segments_:
            segments.append(urljoin(m3u8_url, segment.uri)) # Get the full URL path to segment

        logger.debug(f"Fetched {len(segments)} segments from m3u8 URL")
        return segments

    def download(self, video, quality: str, downloader: str, path: str, callback=None) -> None:
        """
        :param video:
        :param callback:
        :param downloader:
        :param quality:
        :param path:
        :return:
        """

        if callback is None:
            callback = Callback.text_progress_bar

        if downloader == "default":
            self.default(video=video, quality=quality, path=path, callback=callback)

        elif downloader == "threaded":
            threaded_download = self.threaded(max_workers=20, timeout=10)
            threaded_download(self, video=video, quality=quality, path=path, callback=callback)

        elif downloader == "FFMPEG":
            self.FFMPEG(video=video, quality=quality, path=path, callback=callback)

    @classmethod
    def download_segment(cls, url: str, timeout: int) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment, retrying on failure.
        Returns a tuple of the URL, content (empty if failed after retries), and a success flag.
        """

        content = BaseCore().fetch(url, timeout=timeout, get_bytes=True)
        return url, content, True  # Success

    def threaded(self, max_workers: int = 20, timeout: int = 10):
        """
        Creates a wrapper function for the actual download process, with retry logic.
        """

        def wrapper(self, video, quality: str, callback, path: str):
            """
            Download video segments in parallel, with retries for failures, and write to a file.
            """
            segments = self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)
            length = len(segments)
            completed, successful_downloads = 0, 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_hls_part = {
                    executor.submit(self.download_segment, url, timeout): os.path.basename(url)
                    for url in segments
                }

                for future in as_completed(future_to_hls_part):
                    hls_part = future_to_hls_part[future]
                    try:
                        _, data, success = future.result()
                        completed += 1
                        if success:
                            successful_downloads += 1
                        callback(completed, length)  # Update progress callback
                    except Exception as e:
                        logger.error(f"Error processing segment {hls_part}: {e}")

            # Write only successful segments to the output file
            with open(path, 'wb') as file:
                for segment_url in segments:
                    matched_futures = [
                        future for future, hls_part in future_to_hls_part.items()
                        if hls_part == os.path.basename(segment_url)
                    ]
                    if matched_futures:
                        future = matched_futures[0]
                        try:
                            _, data, success = future.result()
                            if success:
                                file.write(data)
                        except Exception as e:
                            logger.error(f"Exception writing segment {segment_url}: {e}")

        return wrapper

    def default(self, video, quality, callback, path, start: int = 0) -> bool:
        buffer = b''
        segments = self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)[start:]
        length = len(segments)

        for i, url in enumerate(segments):
            for _ in range(5):

                segment = self.fetch(url, get_bytes=True)
                buffer += segment
                callback(i + 1, length)
                break

        with open(path, 'wb') as file:
            file.write(buffer)

        return True

    def FFMPEG(self, video, quality: str, callback, path: str) -> bool:
        base_url = video.m3u8_base_url
        new_segment = self.get_m3u8_by_quality(quality=quality, m3u8_url=base_url)
        url_components = base_url.split('/')
        url_components[-1] = new_segment
        new_url = '/'.join(url_components)

        # Build the command for FFMPEG as a list directly
        command = [
            consts.FFMPEG_PATH,
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

        except httpx.RequestError as e:
            logger.error(f"Request failed for URL {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False

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
