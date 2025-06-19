import random

class RuntimeConfig:
    def __init__(self):
        self.max_cache_items = 200
        self.max_retries = 4
        self.request_delay = 0
        self.timeout = 20
        self.ffmpeg_path = "ffmpeg"
        self.proxy = None
        self.verify_ssl = True
        self.user_agents = [
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.3",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.3",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0."
        ]
        self.cookies = None
        self.headers = {
            "User-Agent": random.choice(self.user_agents)
        }

    def rotate_user_agent(self):
        self.headers["User-Agent"] = random.choice(self.user_agents)

    def as_httpx_kwargs(self):
        return {
            "proxies": self.proxy,
            "verify": self.verify_ssl,
            "headers": self.headers,
            "timeout": self.timeout
        }

# Singleton instance needed for my Porn Fetch project
config = RuntimeConfig()