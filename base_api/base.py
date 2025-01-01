import os
import m3u8
import time
import random
import string
import logging
import requests
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
        self.session = requests.Session()
        self.session.headers.update(consts.HEADERS)

    def update_user_agent(self):
        """Updates the User-Agent"""
        self.session.headers.update({"User-Agent": random.choice(consts.USER_AGENTS)})

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

    def fetch(self, url: str, get_bytes: bool = False, stream: bool = False, timeout: int = consts.TIMEOUT,
              get_response: bool = False, save_cache: bool = True, cookies: dict = None,
              allow_redirects: bool = True) -> Union[bytes, str, requests.Response]:
        """
        Fetches content in UTF-8 Text, Bytes or as a stream using multiple request attempts, support for proxies
        and custom timeout.
        """
        content = cache.handle_cache(url)
        if content is not None:
            logger.info(f"Fetched content for: {url} from cache!")
            return content

        for attempt in range(1, consts.MAX_RETRIES):
            if self.total_requests % 3 == 0:  # Change user agent after 3 requests to prevent bot detection
                self.update_user_agent()

            try:
                self.enforce_delay()
                if consts.USE_PROXIES:
                    verify = False  # Disable SSL verification
                    url = url.replace("https://", "http://")  # Replace https to http to support HTTP proxies

                else:
                    verify = True

                response = self.session.get(url, stream=stream, proxies=consts.PROXIES, verify=verify,
                                            timeout=timeout, allow_redirects=allow_redirects, cookies=cookies)

                if response.status_code != 200:
                    logger.error(f"Unexpected status code {response.status_code} for URL: {url}")

                    if response.status_code == 404:
                        logger.error("Website returned 404. Not an issue for xivdeos, but for all other sites")
                        return response # Needed for xvideos. Means that video is unavailable lol

                    continue

                if get_response:
                    return response

                self.total_requests += 1

            except Exception:
                error = traceback.format_exc()
                logger.error(f"Could not fetch: {url} ->: {error}")
                continue  # Next attempt

            logger.debug(f"[{attempt}|{consts.MAX_RETRIES}] Fetch ->: {url}")

            logger.debug(f"[{attempt}|{consts.MAX_RETRIES}] Fetch ->: 200 | Success")
            if get_bytes is False:
                content = response.content.decode("utf-8")
                if save_cache:
                    logger.debug(f"Trying to save content of: {url} in local cache...")
                    cache.save_cache(url, content)

                return content

            return response.content

        raise ConnectionError(f"Could not fetch content for: {url}")

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

        response = BaseCore().fetch(url, timeout=timeout, get_response=True)
        response.raise_for_status()
        return url, response.content, True  # Success

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
                # Map the last part of the URL (filename) to the future
                future_to_hls_part = {executor.submit(self.download_segment, url, timeout): os.path.basename(url) for
                                      url in
                                      segments}

                for future in as_completed(future_to_hls_part):
                    hls_part = future_to_hls_part[future]
                    try:
                        _, data, success = future.result()
                        completed += 1
                        if success:
                            successful_downloads += 1
                        callback(completed, length)  # Update callback regardless of success to reflect progress
                    except Exception as e:
                        raise e

            # Writing only successful downloads to the file
            with open(path, 'wb') as file:
                for segment_url in segments:
                    # Find the future object using the HLS part of the URL as the key
                    matched_futures = [future for future, hls_part in future_to_hls_part.items() if
                                       hls_part == os.path.basename(segment_url)]
                    if matched_futures:
                        logging.info("Got a match")
                        future = matched_futures[0]  # Assuming unique HLS parts, take the first match
                        try:
                            _, data, success = future.result()
                            if success:
                                file.write(data)
                        except Exception:
                            error = traceback.format_exc()
                            logger.error(f"Exception while downloading segment: {error}")

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
            "-c", "copy",  # Copy streams without reencoding
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

    def legacy_download(self, stream: bool, path: str, url: str, callback=None) -> bool:
        response = self.fetch(url, stream=stream, get_bytes=True, get_response=True)
        file_size = int(response.headers.get('content-length', 0))

        if callback is None:
            progress_bar = Callback()

        downloaded_so_far = 0

        if not os.path.exists(path):
            with open(path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    downloaded_so_far += len(chunk)

                    if callback:
                        callback(downloaded_so_far, file_size)

                    else:
                        progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=file_size)

            if not callback:
                del progress_bar

            return True

        else:
            raise FileExistsError("The file already exists.")

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
