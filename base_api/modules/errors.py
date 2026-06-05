# This file contains all custom exceptions for base api. They should be handled by each API individually.
from typing import Optional
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


class KillSwitch(Exception):
    """
    Raised when a kill switch is triggered due to an IP Leak.
    """
    def __init__(self, message: str) -> None:
        self.message = message



class InvalidProxy(Exception):
    """
    Raised when a proxy set by the user is invalid. A basic scheme with regular expressions will check for possible
    proxy configurations such as http, https and socks5 and if the entered proxy follows the certain scheme.
    If it doesn't this exception will be raised.
    """
    def __init__(self, message: str) -> None:
        self.message = message


class UnknownError(Exception):
    """
    Raised when an unknown error occurs that I don't know about yet.
    """
    def __init__(self, message: str) -> None:
        self.message = message


class DownloadCancelled(Exception):
    """
    Raised when a download is cancelled via a stop flag/event.
    """
    def __init__(self, message: str) -> None:
        self.message = message


class SegmentError(Exception):
    """
    Raises when a segment fails to get processed. I never that happen, but you never know.
    """
    def __init__(self, message: str) -> None:
        self.message = message


class NetworkingError(Exception):
    """
    Raises for all general network errors that are usually the fault of the user's internet connection.
    """
    def __init__(self, message: str) -> None:
        self.message = message


class ProxySSLError(Exception):
    """
    Raises if a proxy request fails due to self-signed certificates or invalid TLS verification
    """
    def __init__(self, message: str) -> None:
        self.message = message


class ResourceGone(Exception):
    """
    Raises if a resource is gone (http 410 error)
    """
    def __init__(self, message: str) -> None:
        self.message = message


class BotProtectionDetected(Exception):
    """Raised when Cloudflare or similar bot protection is detected."""
    pass


class SecurityAbort(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        self.message = message_security