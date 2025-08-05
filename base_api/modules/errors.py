# This file contains all custom exceptions for base api. They should be handled by each API individually.

class KillSwitch(Exception):
    """
    Raised when a kill switch is triggered due to an IP Leak.
    """
    def __init__(self, message):
        self.message = message



class InvalidProxy(Exception):
    """
    Raised when a proxy set by the user is invalid. A basic scheme with regular expressions will check for possible
    proxy configurations such as http, https and socks5 and if the entered proxy follows the certain scheme.
    If it doesn't this exception will be raised.
    """
    def __init__(self, message):
        self.message = message


class UnknownError(Exception):
    """
    Raised when an unknown error occurs that I don't know about yet.
    """
    def __init__(self, message):
        self.message = message


class SegmentError(Exception):
    """
    Raises when a segment fails to get processed. I never that happen, but you never know.
    """
    def __init__(self, message):
        self.message = message


class NetworkingError(Exception):
    """
    Raises for all general network errors that are usually the fault of the user's internet connection.
    """
    def __init__(self, message):
        self.message = message


class ProxySSLError(Exception):
    """
    Raises if a proxy request fails due to self-signed certificates or invalid TLS verification
    """
    def __init__(self, message):
        self.message = message


class ResourceGone(Exception):
    """
    Raises if a resource is gone (http 410 error)
    """
    def __init__(self, message):
        self.message = message


class BotProtectionDetected(Exception):
    """Raised when Cloudflare or similar bot protection is detected."""
    pass