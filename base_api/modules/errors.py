# This file contains all custom exceptions for base api. They should be handled by each API individually.
message_security = """
Hey, please stop before proceeding and READ this text:

While solving a Bot Protection challenge Porn Fetch encountered illegal characters
in the extracted challenge code. 

To actually solve the challenge I need to use the exec function in Python which is a big
security risk. A remote attacker that hijacks PornHub or a local hacker that redirects
your DNS queries and serves their own page (there are endless possibilities) could hijack your
system and take full control over your current system account.

Porn Fetch basically strips out ALL possible ways of using this to hack you. Including all 
escape sequences and I also disable all builtin functions while executing, so
a hacker can't randomly open files or import code. 


Now, what happened is, that I detected illegal chars in the response code.
There are 2 possible scenarios:


1) PornHub just randomly changed their challenge page and it now contains different strings
that I need to update and whitelist

2) An actual hacker is trying to hack you right now using an intercepted PornHub
page over fake DNS queries, social engineering or whatever...




Instead of bypassing this yourself please immediately go to GitHub
and open an issue on:

https://github.com/echteralsfake/eaf_base_api/issues

AND: Write an E-Mail to `EchterAlsFakeBS@proton.me`


I take this absolutely serious!
When it comes to your security I take ZERO risks."""




class UnknownError(Exception):
    """
    Raised when an unknown error occurs that I don't know about yet.
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# Networking Errors


class ResourceGone(Exception):
    """
    Raises if a resource is gone (http 410 error)
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class BaseScraperError(Exception):
    ...


class NetworkRequestError(BaseScraperError):
    ...


class HTTPStatusError(BaseScraperError):
    def __init__(self, message: str, status_code: int, url: str):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class RateLimitError(HTTPStatusError):
    def __init__(self, message: str, url: str, retry_after: int = 0):
        super().__init__(message, 429, url)
        self.retry_after = retry_after


class ProxySSLError(Exception):
    """
    Raises if a proxy request fails due to self-signed certificates or invalid TLS verification
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidProxy(Exception):
    """
    Raised when a proxy set by the user is invalid. A basic scheme with regular expressions will check for possible
    proxy configurations such as http, https and socks5 and if the entered proxy follows the certain scheme.
    If it doesn't this exception will be raised.
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)




# Scraping / Processing Errors

class BotProtectionDetected(Exception):
    """Raised when Cloudflare or similar bot protection is detected."""
    pass


class DownloadCancelled(BaseScraperError):
    """
    Raised when a download is canceled via a stop flag/event.
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class SegmentError(Exception):
    """
    Raises when a segment fails to get processed. I never that happen, but you never know.
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class VideoFetchError(BaseScraperError):
    """
    Raised or yielded when a video fails to fetch during concurrent fetching.
    """
    def __init__(self, url: str, original_error: Exception) -> None:
        self.url = url
        self.original_error = original_error
        super().__init__(f"Failed to fetch video at {url}: {original_error}")


class PageFetchError(BaseScraperError):
    def __init__(self, url: str, original_error: Exception) -> None:
        self.url = url
        self.original_error = original_error
        super().__init__(f"Failed to fetch page at {url}: {original_error}")


class ChallengeRegexError(BaseScraperError):
    ...


class ChallengeMathError(BaseScraperError):
    ...


class CallbackError(BaseScraperError):
    def __init__(self, msg: str):
        self.msg = msg


class AccessDeniedError(BaseScraperError):
    ...


class SecurityAbort(ChallengeMathError):
    def __init__(self) -> None:
        self.message = message_security


class PlaylistExtractionError(BaseScraperError):
    pass


class StateLoadError(BaseScraperError):
    pass


class MaxRetriesExceeded(BaseScraperError):
    pass
