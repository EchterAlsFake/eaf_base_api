import asyncio

from dataclasses import dataclass
from typing import Dict, Any, Callable, Literal

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


# Singleton instance needed for my Porn Fetch project
config = RuntimeConfig()
