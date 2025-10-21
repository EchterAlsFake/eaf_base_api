class RuntimeConfig:
    def __init__(self):
        self.max_cache_items = 200
        self.max_retries = 4
        self.request_delay = 0
        self.timeout = 20
        self.max_bandwidth_mb = None # Set speed limit in megabytes per second e.g, 2.0, 3.5 etc...
        self.ffmpeg_path = "ffmpeg"
        self.proxy = None
        self.raise_bot_protection = False
        self.verify_ssl = True
        self.cookies = None
        self.locale = "en-US,en;q=0.9" # If you override this, it could change regexes and thus make stuff not work...
        self.use_http2 = True

# Singleton instance needed for my Porn Fetch project
config = RuntimeConfig()