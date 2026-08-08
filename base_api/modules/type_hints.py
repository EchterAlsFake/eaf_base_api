from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable
from enum import StrEnum
import random

# Download Report is the report the function returns
@dataclass
class DownloadState:
    version: int
    created_at: Any
    updated_at: Any
    m3u8_url: str | None
    quality: str | int
    output_path: Path | str
    segment_dir: Path | str | None
    segment_index_width: int
    start_segment: int
    total: int
    missing: list[int]
    segments: list[str]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# Download state is used for the literal file that tracks it
@dataclass
class DownloadReport:
    status: str
    total: int
    downloaded: int
    missing: list[int]
    missing_urls: list[str]
    segment_dir: Path | str | None
    segment_state_path: Path | str | None
    start_segment: int
    quality: str | int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ResultOrder(StrEnum):
    """Controls when ``Helper`` exposes completed item results."""

    COMPLETION = "completion"
    ORIGINAL = "original"


class ErrorMode(StrEnum):
    """Terminal action after retries are exhausted."""

    RAISE = "raise"
    YIELD = "yield"
    SKIP = "skip"


class ErrorAction(StrEnum):
    """Decision optionally returned by a user-provided error handler."""

    RETRY = "retry"
    RAISE = "raise"
    YIELD = "yield"
    SKIP = "skip"


class ScrapeStage(StrEnum):
    """Identifies whether a yielded failure belongs to a page or an item."""

    PAGE = "page"
    ITEM = "item"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    Bounded exponential retry configuration for one Helper stage.

    ``max_attempts`` includes the first call.  The default performs one attempt,
    which avoids duplicating retries already performed by ``BaseCore.fetch``.
    ``jitter`` adds a uniformly random number of seconds to each retry delay.
    """

    max_attempts: int = 1
    base_delay: float = 0.0
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.0
    retry_for: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("RetryPolicy delays and jitter cannot be negative")
        if self.multiplier < 1:
            raise ValueError("RetryPolicy.multiplier must be at least 1")
        if not self.retry_for or not all(
            isinstance(item, type) and issubclass(item, Exception)
            for item in self.retry_for
        ):
            raise TypeError("RetryPolicy.retry_for must contain Exception classes")

    def permits(self, error: Exception) -> bool:
        """Return whether this exception type is eligible for automatic retry."""
        return isinstance(error, self.retry_for)

    def delay_after(self, attempt: int) -> float:
        """Return the delay after the numbered failed attempt."""
        exponential = self.base_delay * (self.multiplier ** max(attempt - 1, 0))
        return min(exponential, self.max_delay) + random.uniform(0.0, self.jitter)


@dataclass(frozen=True, slots=True)
class ScrapeErrorContext:
    """Complete context passed to a page or item error handler."""

    stage: ScrapeStage
    url: str
    error: Exception
    attempt: int
    max_attempts: int
    page_index: int
    item_index: int | None


type ErrorHandler = Callable[
    [ScrapeErrorContext], ErrorAction | Awaitable[ErrorAction]
]
