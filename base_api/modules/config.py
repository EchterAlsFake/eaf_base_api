class RuntimeConfig:
    def __init__(self):
        self.max_cache_items = 200
        self.max_retries = 4
        self.request_delay = 0
        self.timeout = 20
        self.max_bandwidth_mb = None # Set speed limit in megabytes per second e.g, 2.0, 3.5 etc...
        self.proxies = None
        self.http_version = "v3" # "v3 = HTTP/3.0, v2 = HTTP/2.0, v1 = HTTP/1.1
        self.dns_over_https = None
        self.impersonation = "chrome"
        self.custom_ja3 = None # Absolutely only for advanced users, research before you use this!!!
        self.proxy_auth = None
        self.verify_ssl = True
        self.trust_env = False
        self.cookies = None
        self.locale = "en-US,en;q=0.9" # If you override this, it could change regexes and thus make stuff not work...
        self.max_workers_download = 20
        self.videos_concurrency = 5
        self.pages_concurrency = 2


# Singleton instance needed for my Porn Fetch project
config = RuntimeConfig()