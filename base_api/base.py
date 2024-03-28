import requests
import logging
import time

from base_api.modules.quality import Quality
from base_api.modules.progress_bars import Callback
from base_api.modules.download import default, threaded, FFMPEG
from base_api.modules.consts import MAX_RETRIES
from pathlib import Path


def setup_api(do_logging=False):
    if do_logging:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    else:
        logging.disable(logging.CRITICAL)


base_qualities = ["250p", "360p", "480p", "720p", "1080p", "1440p", "2160p"]


class Core:
    @classmethod
    def get_content(cls, url, headers=None, cookies=None):
        for i in range(MAX_RETRIES):
            try:
                logging.debug(f"Trying to fetch {url} / Attempt: [{i+1}]")
                response = requests.get(url, headers=headers, cookies=cookies)
                if response.status_code == 200:
                    logging.info(f"Successfully fetched {url} on attempt [{i+1}]")
                    return response.content
                else:
                    logging.warning(f"Failed to fetch {url} with status code {response.status_code} on attempt [{i+1}]")
            except requests.exceptions.RequestException as e:
                logging.error(f"Exception occurred when trying to fetch {url} on attempt [{i+1}]: {e}")
            if i < MAX_RETRIES - 1:  # Implement exponential backoff
                time.sleep(2 ** i)
        logging.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts.")
        logging.error("Returning None")
        return None

    @classmethod
    def fix_quality(cls, quality):
        """This method gives the user the opportunity to pass a string instead of the quality object"""
        if isinstance(quality, Quality):
            return quality

        elif str(quality) == "best":
            return Quality.BEST

        elif str(quality) == "half":
            return Quality.HALF

        elif str(quality) == "worst":
            return Quality.WORST

    def get_available_qualities(self, m3u8_base_url):
        """Returns the available qualities from the M3U8 base url"""
        content = self.get_content(m3u8_base_url).decode("utf-8")
        lines = content.splitlines()

        quality_url_map = {}

        for line in lines:
            for quality in base_qualities:
                if f"hls-{quality}" in line:
                    quality_url_map[quality] = line

        self.quality_url_map = quality_url_map
        self.available_qualities = list(quality_url_map.keys())
        return self.available_qualities

    def get_m3u8_by_quality(self, quality, m3u8_base_url):
        """Returns the m3u8 url for the given quality"""
        quality = self.fix_quality(quality)

        self.get_available_qualities(m3u8_base_url)
        if quality == Quality.BEST:
            selected_quality = max(self.available_qualities, key=lambda q: base_qualities.index(q))
        elif quality == Quality.WORST:
            selected_quality = min(self.available_qualities, key=lambda q: base_qualities.index(q))
        elif quality == Quality.HALF:
            sorted_qualities = sorted(self.available_qualities, key=lambda q: base_qualities.index(q))
            middle_index = len(sorted_qualities) // 2
            selected_quality = sorted_qualities[middle_index]

        return self.quality_url_map.get(selected_quality)

    def download(self, video, quality, downloader, path, callback=None):
        """
        :param callback:
        :param downloader:
        :param quality:
        :param path:
        :return:
        """
        quality = self.fix_quality(quality)

        if callback is None:
            callback = Callback.text_progress_bar

        if downloader == default or str(downloader) == "default":
            default(video=video, quality=quality, path=path, callback=callback)

        elif downloader == threaded or str(downloader) == "threaded":
            threaded_download = threaded(max_workers=20, timeout=10)
            threaded_download(video=video, quality=quality, path=path, callback=callback)

        elif downloader == FFMPEG or str(downloader) == "FFMPEG":
            FFMPEG(video=video, quality=quality, path=path, callback=callback)

    @classmethod
    def correct_path(cls, path):
        return Path(path)

    @classmethod
    def get_segments(cls, quality, m3u8_base_url):
        quality = Core().fix_quality(quality)
        base_url = m3u8_base_url
        new_segment = Core().get_m3u8_by_quality(quality, m3u8_base_url=base_url)
        # Split the base URL into components
        url_components = base_url.split('/')

        # Replace the last component with the new segment
        url_components[-1] = new_segment

        # Rejoin the components into the new full URL
        new_url = '/'.join(url_components)
        master_src = Core().get_content(url=new_url).decode("utf-8")

        urls = [l for l in master_src.splitlines()
                if l and not l.startswith('#')]

        segments = []

        for url in urls:
            url_components[-1] = url
            new_url = '/'.join(url_components)
            segments.append(new_url)

        return segments

    @classmethod
    def strip_title(cls, title: str) -> str:
        """
        :param title:
        :return: str: strips out non UTF-8 chars of the title
        """
        illegal_chars = '<>:"/\\|?*'
        cleaned_title = ''.join([char for char in title if char in string.printable and char not in illegal_chars])
        return cleaned_title