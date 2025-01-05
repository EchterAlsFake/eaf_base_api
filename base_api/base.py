import os
import time
import m3u8
import httpx
import random
import string
import asyncio
import logging
import traceback
import time
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
        self.total_requests = 0
        self.user_agent = random.choice(consts.USER_AGENTS)
        self.session = httpx.AsyncClient(
            proxy=consts.PROXY if consts.PROXY else None,
            verify=not consts.PROXY,
            headers=consts.HEADERS
        )

    async def enforce_delay(self):
        delay = consts.REQUEST_DELAY
        if delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                logger.debug(f"Enforcing delay of {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)

    async def fetch(
            self,
            url: str,
            get_bytes: bool = False,
            timeout: int = consts.TIMEOUT,
            get_response: bool = False,
            save_cache: bool = True,
            cookies: dict = None,
            allow_redirects: bool = True,
    ) -> Union[bytes, str, httpx.Response, None]:
        # Check cache
        content = cache.handle_cache(url)
        if content is not None:
            logger.info(f"Fetched content for: {url} from cache!")
            return content

        for attempt in range(1, consts.MAX_RETRIES + 1):
            try:
                if self.total_requests % 3 == 0:
                    self.user_agent = random.choice(consts.USER_AGENTS)

                await self.enforce_delay()

                response = await self.session.get(
                    url,
                    timeout=timeout,
                    cookies=cookies,
                    follow_redirects=allow_redirects
                )
                self.total_requests += 1

                if response.status_code != 200:
                    logger.warning(
                        f"Attempt {attempt}: Unexpected status code {response.status_code} for URL: {url}"
                    )
                    if response.status_code == 404:
                        logger.error("Resource not found (404). This may indicate the content is unavailable.")
                        return None
                    continue

                logger.debug(f"Attempt {attempt}: Successfully fetched URL: {url}")

                if get_response:
                    return response

                content = await response.aread() if get_bytes else response.text
                if save_cache:
                    cache.save_cache(url, content)
                return content

            except httpx.RequestError as e:
                logger.error(f"Attempt {attempt}: Request error for URL {url}: {e}")

            except Exception as e:
                logger.error(f"Attempt {attempt}: Unexpected error for URL {url}: {traceback.format_exc()}")

            logger.info(f"Retrying ({attempt}/{consts.MAX_RETRIES}) for URL: {url}")

        logger.error(f"Failed to fetch URL {url} after {consts.MAX_RETRIES} attempts.")
        return None

    async def close(self):
        await self.session.aclose()

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
    async def get_m3u8_by_quality(self, m3u8_url: str, quality: str) -> str:
        """Fetches the m3u8 URL for a given quality by extracting all possible sub-m3u8 URLs from the primary
        m3u8 playlist"""
        playlist_content = await self.fetch(url=m3u8_url)
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
    async def get_segments(self, m3u8_url_master: str, quality: str) -> list:
        """Gets all video segments from a given quality and the primary m3u8 URL"""
        m3u8_url = await self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        logger.debug(f"Trying to fetch segment from m3u8 ->: {m3u8_url}")
        content = await self.fetch(url=m3u8_url)
        segments_ = m3u8.loads(content).segments
        segments = []

        for segment in segments_:
            segments.append(urljoin(m3u8_url, segment.uri)) # Get the full URL path to segment

        logger.debug(f"Fetched {len(segments)} segments from m3u8 URL")
        return segments

    async def download(self, video, quality: str, downloader: str, path: str, callback=None) -> None:
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
            await self.default(video=video, quality=quality, path=path, callback=callback)

        elif downloader == "threaded":
            await self.threaded(video=video, quality=quality, path=path, callback=callback)

        elif downloader == "FFMPEG":
            await self.FFMPEG(video=video, quality=quality, path=path, callback=callback)

    async def download_segment(self, url: str) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment.
        Returns a tuple of the URL, content (empty if failed), and a success flag.
        """
        try:
            response = await self.fetch(url=url, get_response=True)
            response.raise_for_status()
            content = await response.aread()
            return url, content, True  # Success

        except Exception as e:
            logger.error(f"Failed to download segment {url}: {e}")
            return url, b'', False  # Failure

    async def threaded(self, video, quality: str, callback, path: str, max_concurrent_downloads: int = 20):
        """
        Download video segments asynchronously in parallel, write them to a file, and handle retries.
        The `callback` function is called after each segment is processed.
        """
        segments = await self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)
        total_segments = len(segments)
        completed = 0
        successful_downloads = 0

        semaphore = asyncio.Semaphore(max_concurrent_downloads)
        segment_data_list = [None] * total_segments  # To store segment data in order

        async def process_segment(index, url):
            nonlocal completed, successful_downloads
            async with semaphore:
                url, data, success = await self.download_segment(url)
                completed += 1

                if success:
                    successful_downloads += 1
                    segment_data_list[index] = data  # Store data at the correct index
                else:
                    segment_data_list[index] = b''  # Empty data for failed downloads

                # Call the callback function with progress
                if callback:
                    callback(completed, total_segments)

        # Create tasks for all segments
        tasks = [
            asyncio.create_task(process_segment(idx, segment_url))
            for idx, segment_url in enumerate(segments)
        ]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Write segments to the file in order
        with open(path, 'wb') as file:
            for data in segment_data_list:
                file.write(data or b'')

        logger.info(f"Downloaded {successful_downloads}/{total_segments} segments successfully.")
        return successful_downloads, total_segments

    async def default(self, video, quality, callback, path, start: int = 0) -> bool:
        buffer = b''
        segments = await self.get_segments(quality=quality, m3u8_url_master=video.m3u8_base_url)[start:]
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

    async def FFMPEG(self, video, quality: str, callback, path: str) -> bool:
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

    async def legacy_download(self, path: str, url: str, callback=None) -> bool:
        """
        Download a file using streaming, with support for progress updates.
        """
        try:
            async with self.session.stream("GET", url, timeout=30) as response:  # Use a reasonable timeout
                response.raise_for_status()

                file_size = int(response.headers.get('content-length', 0))

                if callback is None:
                    progress_bar = Callback()

                downloaded_so_far = 0

                # Open file for writing
                with open(path, 'wb') as file:
                    async for chunk in response.aiter_bytes(chunk_size=1024):
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
