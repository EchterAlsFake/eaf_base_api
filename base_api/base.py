import m3u8
import time
import random
import string
import logging
import requests
import traceback

from modules import consts
from urllib.parse import urljoin
from modules.download import default, threaded, FFMPEG
from modules.progress_bars import Callback
"""
from modules import download
from modules import progress_bars
"""
logging.basicConfig(format='%(name)s %(levelname)s %(asctime)s %(message)s', datefmt='%I:%M:%S %p')
logger = logging.getLogger("BASE API")
logger.setLevel(logging.DEBUG)



class BaseCore:
    def __init__(self):
        self.last_request_time = time.time()
        self.headers = {
            "User-Agent": random.choice(consts.USER_AGENTS),
        }
        self.total_requests = 0 # Tracks how many requests have been made
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def update_user_agent(self):
        self.headers.update({"User-Agent": random.choice(consts.USER_AGENTS)})

    def enforce_delay(self):
        delay = consts.REQUEST_DELAY
        if delay > 0:
            time_since_last_request = time.time() - self.last_request_time
            logger.debug(f"Time since last request: {time_since_last_request:.2f} seconds.")
            if time_since_last_request < delay:
                sleep_time = delay - time_since_last_request
                logger.debug(f"Enforcing delay of {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

    def get_content(self, url, get_bytes=False, stream=False, timeout=consts.TIMEOUT):
        for attempt in range(1, consts.MAX_RETRIES):
            if self.total_requests % 3 == 0:
                self.update_user_agent()

            try:
                self.enforce_delay()
                if consts.USE_PROXIES:
                    verify = False
                    url = url.replace("https://", "http://")

                else:
                    verify = True

                response = self.session.get(url, stream=stream, proxies=consts.PROXIES, verify=verify)
                self.total_requests += 1

            except Exception:
                error = traceback.format_exc()
                logger.error(f"Could not fetch: {url} ->: {error}")
                continue

            logger.debug(f"[{attempt}|{consts.MAX_RETRIES}] Fetch ->: {url}")

            if response.status_code == 200:
                logger.debug(f"[{attempt}|{consts.MAX_RETRIES}] Fetch ->: 200 | Success")
                if get_bytes is False:
                    return response.content.decode("utf-8")

                return response.content

            else:
                logger.error(f"Received unexpected status code -->: {response.status_code}")

    @classmethod
    def strip_title(cls, title: str) -> str:
        """
        :param title:
        :return: str: strips out non UTF-8 chars of the title
        """
        illegal_chars = '<>:"/\\|?*'
        cleaned_title = ''.join([char for char in title if char in string.printable and char not in illegal_chars])
        return cleaned_title

    def get_m3u8_by_quality(self, m3u8_url: str, quality: str):
        playlist_content = self.get_content(url=m3u8_url)
        playlist = m3u8.loads(playlist_content)
        logger.debug(f"Resolved m3u8 playlist: {m3u8_url}")

        if not playlist.is_variant:
            raise ValueError("Provided URL is not a master playlist.")

        # Extract available qualities and URLs
        qualities = {
            (variant.stream_info.resolution or "unknown"): variant.uri
            for variant in playlist.playlists
        }

        # Sort qualities by resolution (width x height)
        sorted_qualities = sorted(
            qualities.items(),
            key=lambda x: (x[0][0] * x[0][1]) if x[0] != "unknown" else 0
        )

        # Select based on quality preference
        if quality == "best":
            url_tmp = sorted_qualities[-1][1]  # Highest resolution URL

        elif quality == "worst":
            url_tmp = sorted_qualities[0][1]  # Lowest resolution URL

        elif quality == "half":
            url_tmp = sorted_qualities[len(sorted_qualities) // 2][1]  # Mid-quality URL

        else:
            raise ValueError("Invalid quality provided.")

        full_url = urljoin(m3u8_url, url_tmp)
        return full_url

    def get_segments(self, m3u8_url_master, quality):
        m3u8_url = self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        logger.debug(f"Trying to fetch segment from m3u8 ->: {m3u8_url}")
        content = self.get_content(url=m3u8_url)
        segments = m3u8.loads(content).segments
        logger.debug(f"Fetched {len(segments)} segments from m3u8 URL")
        return segments

    def download(self, video, quality, downloader, path, callback=None):
        """
        :param callback:
        :param downloader:
        :param quality:
        :param path:
        :return:
        """

        if callback is None:
            callback = Callback.text_progress_bar

        if downloader == "default":
            default(video=video, quality=quality, path=path, callback=callback)

        elif downloader == "threaded":
            threaded_download = threaded(max_workers=20, timeout=10)
            threaded_download(video=video, quality=quality, path=path, callback=callback)

        elif downloader == "FFMPEG":
            FFMPEG(video=video, quality=quality, path=path, callback=callback)


core = BaseCore()