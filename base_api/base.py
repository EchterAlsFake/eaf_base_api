import requests
import logging
import time

from base_api.modules.quality import Quality
from base_api.modules.progress_bars import Callback
from base_api.modules.download import default, threaded, FFMPEG

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING)
base_qualities = ["250p", "360p", "480p", "720p", "1080p", "1440p", "2160p"]


class Core:
    @classmethod
    def get_content(cls, url, headers=None, cookies=None, retries=4):
        for i in range(retries):
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
            if i < retries - 1:  # Implement exponential backoff
                time.sleep(2 ** i)
        logging.error(f"Failed to fetch {url} after {retries} attempts.")
        logging.error("Returning None")
        return None

    @classmethod
    def fix_quality(cls, quality):
        """This method gives the user the opportunity to pass a string instead of the quality object"""
        logging.info(f"Quality type is: {quality}")
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
        logging.info(f"Selected Quality: {quality}")

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

    def download(self, video, downloader, quality, output_path, callback=None):
        """
        :param callback:
        :param downloader:
        :param quality:
        :param output_path:
        :return:
        """
        quality = self.fix_quality(quality)

        if callback is None:
            callback = Callback.text_progress_bar

        if downloader == default or str(downloader) == "default":
            default(video=video, quality=quality, path=output_path, callback=callback)

        elif downloader == threaded or str(downloader) == "threaded":
            threaded_download = threaded(max_workers=20, timeout=10)
            threaded_download(video=video, quality=quality, path=output_path, callback=callback)

        elif downloader == FFMPEG or str(downloader) == "FFMPEG":
            FFMPEG(video=video, quality=quality, path=output_path, callback=callback)
