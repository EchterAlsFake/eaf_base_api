from typing import Dict, Optional


class RuntimeConfig:
    def __init__(self) -> None:
        self.max_cache_items: int = 200
        self.max_retries: int = 4
        self.request_delay: int = 0
        self.timeout: int = 20
        self.max_bandwidth_mb: Optional[float] = None # Set speed limit in megabytes per second e.g, 2.0, 3.5 etc...
        self.proxies: Optional[Dict[str, str]] = None
        self.http_version: str = "v3" # "v3 = HTTP/3.0, v2 = HTTP/2.0, v1 = HTTP/1.1
        self.dns_over_https: Optional[str] = None
        self.impersonation: str = "chrome"
        self.custom_ja3: Optional[str] = None # Absolutely only for advanced users, research before you use this!!!
        self.proxy_auth: Optional[str] = None
        self.verify_ssl: bool = True
        self.trust_env: bool = False
        self.cookies: Optional[Dict[str, str]] = None
        self.locale: str = "en-US,en;q=0.9" # If you override this, it could change regexes and thus make stuff not work...
        self.max_workers_download: int = 20
        self.videos_concurrency: int = 5
        self.pages_concurrency: int = 2


# Singleton instance needed for my Porn Fetch project
config = RuntimeConfig()