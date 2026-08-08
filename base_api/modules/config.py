import asyncio

from dataclasses import dataclass
from typing import Dict, Any, Callable, Literal

from mypy.applytype import Iterable

from base_api import ResultOrder, ErrorMode, RetryPolicy, ErrorHandler

type callback_hint = Callable[[int, int], None] | None
type possible_qualities = Literal["hd", "sd", "144p", "240p", "360p", "480p", "540p", "720p", "1080p", "1440p", "2160p",
                                   "best", "worst", "half"]

type possible_qualities_int = Literal[144, 240, 360, 480, 540, 720, 1080, 1440, 2160]


class RuntimeConfig:
    def __init__(self) -> None:
        self.response_cache_size_bytes: int = 32 * 1024 * 1024
        self.response_cache_ttl: float = 300.0
        self.segment_cache_size_bytes: int = 8 * 1024 * 1024
        self.segment_cache_ttl: float = 300.0
        self.request_attempts: int = 4
        self.request_retry_initial_delay: float = 0.5
        self.request_retry_max_delay: float = 30.0
        self.request_retry_jitter: float = 0.5
        self.request_delay: int = 0
        self.timeout: int = 20
        self.max_bandwidth_mb: float| None = None # Set speed limit in megabytes per second e.g, 2.0, 3.5 etc...
        self.proxy = None
        self.http_version: str = "v2" # "v3 = HTTP/3.0, v2 = HTTP/2.0, v1 = HTTP/1.1
        self.dns_over_https: str | None = None
        self.impersonation: str = "chrome"
        self.custom_ja3: str | None = None # Absolutely only for advanced users, research before you use this!!!
        self.proxy_auth: str | None = None
        self.verify_ssl: bool = True
        self.trust_env: bool = False
        self.cookies: Dict[str, str] | None = None
        self.locale: str = "en-US,en;q=0.9" # If you override this, it could change regexes and thus make stuff not work...
        self.max_workers_download: int = 20
        self.videos_concurrency: int = 5
        self.pages_concurrency: int = 2
        self.interface: str | None = None # IP Address of the network interface you want to bind to


@dataclass
class BaseConfigDownload:
    quality: possible_qualities | possible_qualities_int
    path: Any = "./"
    callback: callback_hint = None
    no_title: bool = False
    stop_event: asyncio.Event | None = None

@dataclass
class DownloadConfigHLS(BaseConfigDownload):
    m3u8_base_url: Any = None
    remux: bool = False
    start_segment: int = 0
    segment_state_path: str | None = None
    segment_dir: str | None = None
    return_report: bool = False
    cleanup_on_stop: bool = True
    keep_segment_dir: bool = False
    callback_remux: callback_hint = None
    ios_support: bool = False


@dataclass
class DownloadConfigRAW(BaseConfigDownload):
    allow_multipart: bool = True
    max_workers: int = 5
    read_timeout: float = 120.0
    chunk_size: int = 1024
    max_retries: int = 5


class IteratorConfig:
    max_page_concurrency: int = 5 # How many pages to scrape at max concurrency
    max_item_concurrency: int = 20 # How many items to scrape at max concurrency
    max_pending_items: int | None = None  # Defines the max pending limit which is needed to limit memory usage
    extract_in_thread: bool = True # Whether to offload the extraction to asyncio.to_thread(). Useful for heavy selectolax parsing, useless for simple JSON parsing
    order: ResultOrder | str = ResultOrder.COMPLETION
    page_error_mode: ErrorMode | str = ErrorMode.YIELD # How to handle page errors
    item_error_mode: ErrorMode | str = ErrorMode.YIELD # How to handle item errors
    page_retry: RetryPolicy | None = None
    item_retry: RetryPolicy | None = None
    page_error_handler: ErrorHandler | None = None # Custom function for handling retrying (pages)
    item_error_handler: ErrorHandler | None = None  # Custom function for handling retrying (items)

    load_specific_fields: Iterable[str] = () # Loads the actual fields from the available and fetched sources
    load_specific_sources: Iterable[str] = () # Runs before load_fields(), defines which source to load e.g., from html or from an API endpoint


    _page_request_method: str = "GET"  # Some pages require POST requests (depends per API)
    _item_url_key: str = "url" # The actual Video / Short whatever URL returned in the dictionary by the item extractor

# Singleton instance needed for my Porn Fetch project
config = RuntimeConfig()
