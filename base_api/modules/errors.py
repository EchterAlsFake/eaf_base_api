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
    def __init__(self, message: str, url: str, retry_after: float | None = None):
        super().__init__(message, 429, url)
        self.retry_after = retry_after


class RequestRetriesExhausted(NetworkRequestError):
    """Raised after a retryable request consumes its complete attempt budget."""

    def __init__(self, url: str, attempts: int, last_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Request to {url} failed after {attempts} attempts: {last_error}"
        )


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


class MediaFieldError(BaseScraperError):
    """Base class for errors involving a field on a ``BaseMedia`` model."""


class UnknownMediaFieldError(MediaFieldError, AttributeError):
    """Raised when a caller asks ``BaseMedia`` to load an undeclared field."""

    def __init__(self, model_name: str, field_name: str) -> None:
        self.model_name = model_name
        self.field_name = field_name
        super().__init__(f"{model_name!s} has no dataclass field named {field_name!r}")


class FieldNotLoadableError(MediaFieldError):
    """Raised when a real dataclass field has no loader source assigned to it."""

    def __init__(self, model_name: str, field_name: str) -> None:
        self.model_name = model_name
        self.field_name = field_name
        super().__init__(
            f"{model_name}.{field_name} is not a loadable media field; "
            "declare it with media_field(...) before requesting it"
        )


class DataNotLoadedError(MediaFieldError):
    """
    Raised by direct attribute access when a loadable field is still unresolved.

    ``None`` never causes this exception.  The media implementation uses a private
    sentinel for unresolved fields, so a loader may deliberately return ``None``
    when the remote service does not provide an optional value.
    """

    def __init__(
        self,
        model_name: str,
        field_name: str,
        url: str,
        sources: tuple[str, ...],
        source_errors: dict[str, BaseException] | None = None,
    ) -> None:
        self.model_name = model_name
        self.field_name = field_name
        self.url = url
        self.sources = sources
        self.source_errors = source_errors or {}

        source_list = ", ".join(repr(source) for source in sources)
        message = (
            f"{model_name}.{field_name} has not been loaded for {url!r}. "
            f"Eligible sources: {source_list}. Call "
            f"await media.load_fields({field_name!r}) or "
            f"await media.load_sources({source_list})."
        )
        if self.source_errors:
            failures = ", ".join(
                f"{source}={type(error).__name__}: {error}"
                for source, error in self.source_errors.items()
            )
            message += f" Previous source failures: {failures}."

        super().__init__(message)


class LoaderConfigurationError(BaseScraperError):
    """Raised when a media class declares a source without a usable loader."""


class LoaderContractError(BaseScraperError):
    """Raised when a source loader returns incomplete or unexpected field data."""

    def __init__(self, model_name: str, source: str, url: str, details: str) -> None:
        self.model_name = model_name
        self.source = source
        self.url = url
        self.details = details
        super().__init__(
            f"Loader {model_name}.{source!s} violated its result contract for "
            f"{url!r}: {details}"
        )


class MediaLoadError(BaseScraperError):
    """Wraps an exception raised while one ``BaseMedia`` source was loading."""

    def __init__(self, model_name: str, source: str, url: str, original_error: Exception) -> None:
        self.model_name = model_name
        self.source = source
        self.url = url
        self.original_error = original_error
        super().__init__(
            f"Failed to load source {source!r} for {model_name} at {url}: "
            f"{original_error}"
        )


class MediaLoadErrors(BaseScraperError):
    """Contains all source failures from one multi-source loading request."""

    def __init__(self, errors: tuple[BaseException, ...]) -> None:
        self.errors = errors
        summary = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"Multiple media sources failed: {summary}")


class ScrapeOperationError(BaseScraperError):
    """Base class for a terminal page or item error from ``Helper``."""

    def __init__(
        self,
        message: str,
        *,
        url: str,
        original_error: Exception,
        attempt: int,
        page_index: int,
        item_index: int | None,
    ) -> None:
        self.url = url
        self.original_error = original_error
        self.attempt = attempt
        self.page_index = page_index
        self.item_index = item_index
        super().__init__(message)


class PageFetchError(ScrapeOperationError):
    """A page could not be fetched or its extractor output was invalid."""

    def __init__(self, url: str, original_error: Exception, attempt: int, page_index: int) -> None:
        super().__init__(
            f"Failed to process page {page_index} at {url} on attempt {attempt}: "
            f"{original_error}",
            url=url,
            original_error=original_error,
            attempt=attempt,
            page_index=page_index,
            item_index=None,
        )


class ItemFetchError(ScrapeOperationError):
    """An extracted item could not be constructed or loaded."""

    def __init__(
        self,
        url: str,
        original_error: Exception,
        attempt: int,
        page_index: int,
        item_index: int,
    ) -> None:
        super().__init__(
            f"Failed to process item {page_index}:{item_index} at {url} on "
            f"attempt {attempt}: {original_error}",
            url=url,
            original_error=original_error,
            attempt=attempt,
            page_index=page_index,
            item_index=item_index,
        )


class ErrorHandlerError(BaseScraperError):
    """Raised when a user-provided Helper error handler itself fails."""

    def __init__(self, stage: str, url: str, original_error: Exception) -> None:
        self.stage = stage
        self.url = url
        self.original_error = original_error
        super().__init__(f"The {stage} error handler failed for {url}: {original_error}")


class ChallengeRegexError(BaseScraperError):
    ...


class ChallengeMathError(BaseScraperError):
    ...


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
