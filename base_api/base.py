from __future__ import annotations
import re
import os
import time
import hashlib
import string
import shutil
import random
import asyncio
import inspect
import logging
import traceback
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from functools import lru_cache
from urllib.parse import urljoin
from dataclasses import MISSING, dataclass, field, fields
from curl_cffi import CurlOpt # Used for DNS over HTTPS
from curl_cffi.requests.errors import RequestsError
from curl_cffi.requests import AsyncSession, Response
from cachetools import TTLCache
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter, retry_if_exception, RetryError
from typing import (
    Union, Callable, Tuple, AsyncGenerator, ClassVar, Generic, TypeVar,
    cast, List, Dict, Any, Awaitable, Self, TYPE_CHECKING, Protocol,
)


# 1. Standardize on relative imports
from base_api.modules.errors import *
from base_api.modules.type_hints import (
    DownloadReport, ResultOrder, ErrorMode, ErrorAction, ScrapeStage, 
    RetryPolicy, ScrapeErrorContext, ErrorHandler
)
from base_api.modules.static_functions import (
    load_segment_state, parse_retry_after, log_precondition_failed,
    write_segment_state, build_segment_state, get_segment_index_width,
    segment_file_path, is_video_playlist, height_from_variant,
    pick_by_height, normalize_quality_value,
    parse_challenge, other_challenge, least_factors,
    collect_variants, pick_by_label
)
from base_api.modules.config import config, RuntimeConfig, DownloadConfigHLS, DownloadConfigRAW, IteratorConfig
from base_api.modules.progress_bars import Callback
from base_api.modules.logger import configure_app_logging

# 2. Handle optional dependencies cleanly
try:
    import m3u8
except ImportError:
    m3u8 = None

# 3. Handle specific runtime imports
if TYPE_CHECKING:
    from av.audio.codeccontext import AudioCodecContext
    import m3u8

# The following imports are optional, because they depend on per API and I want to be as memory efficient as possible

try:
    import m3u8
    # Needed for all videos that use HLS streaming. Some do not and use mp4 containers / files instead
except (ModuleNotFoundError, ImportError):
    m3u8 = None  # type: ignore


UA_DESKTOP_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/122.0.0.0 Safari/537.36")

REGEX_CHALLENGE = re.compile(r'var p=(\d+); var s=(\d+);.*?(\d+):1;', re.DOTALL)


class CachePolicy(StrEnum):
    """Control whether a text request reads from or writes to the cache."""

    USE = "use"
    BYPASS = "bypass"
    REFRESH = "refresh"


@dataclass(frozen=True, slots=True)
class RequestCacheKey:
    """Identity of a cacheable HTTP request without retaining credentials."""

    method: str
    url: str
    allow_redirects: bool
    params_fingerprint: str
    body_fingerprint: str
    headers_fingerprint: str
    cookies_fingerprint: str


@dataclass(frozen=True, slots=True)
class SegmentCacheKey:
    master_url: str
    quality: str


class CacheBackend(Protocol):
    """Storage contract consumed by :class:`BaseCore`."""

    def get_response(self, key: RequestCacheKey) -> str | None: ...
    def set_response(self, key: RequestCacheKey, content: str) -> None: ...
    def delete_response(self, key: RequestCacheKey) -> None: ...
    def invalidate_url(self, url: str) -> None: ...
    def get_segments(self, key: SegmentCacheKey) -> list[str] | None: ...
    def set_segments(self, key: SegmentCacheKey, segments: Sequence[str]) -> None: ...


def _text_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _segments_size(value: tuple[str, ...]) -> int:
    return sum(len(segment.encode("utf-8")) for segment in value)


def _freeze_cache_value(value: Any) -> Any:
    """Convert common request values into a deterministic, hashable structure."""
    if isinstance(value, Mapping):
        items = (
            (_freeze_cache_value(key), _freeze_cache_value(item))
            for key, item in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_cache_value(item) for item in value), key=repr))
    if isinstance(value, bytearray):
        return bytes(value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    return repr(value)


def _cache_fingerprint(value: Any) -> str:
    frozen = _freeze_cache_value(value)
    return hashlib.sha256(repr(frozen).encode("utf-8")).hexdigest()


class Cache(CacheBackend):
    """Thread-safe, bounded TTL caches for text responses and HLS segments."""

    def __init__(self, configuration: "RuntimeConfig") -> None:
        self._responses: TTLCache[RequestCacheKey, str] = TTLCache(
            maxsize=max(1, configuration.response_cache_size_bytes),
            ttl=configuration.response_cache_ttl,
            getsizeof=_text_size,
        )
        self._segments: TTLCache[SegmentCacheKey, tuple[str, ...]] = TTLCache(
            maxsize=max(1, configuration.segment_cache_size_bytes),
            ttl=configuration.segment_cache_ttl,
            getsizeof=_segments_size,
        )
        self._responses_enabled = configuration.response_cache_size_bytes > 0
        self._segments_enabled = configuration.segment_cache_size_bytes > 0
        self.lock = threading.RLock()

    def get_response(self, key: RequestCacheKey) -> str | None:
        if not self._responses_enabled:
            return None
        with self.lock:
            return self._responses.get(key)

    def set_response(self, key: RequestCacheKey, content: str) -> None:
        if not self._responses_enabled or _text_size(content) > self._responses.maxsize:
            return
        with self.lock:
            self._responses[key] = content

    def delete_response(self, key: RequestCacheKey) -> None:
        with self.lock:
            self._responses.pop(key, None)

    def invalidate_url(self, url: str) -> None:
        with self.lock:
            for key in tuple(self._responses):
                if key.url == url:
                    self._responses.pop(key, None)

    def get_segments(self, key: SegmentCacheKey) -> list[str] | None:
        if not self._segments_enabled:
            return None
        with self.lock:
            segments = self._segments.get(key)
            return list(segments) if segments is not None else None

    def set_segments(self, key: SegmentCacheKey, segments: Sequence[str]) -> None:
        frozen_segments = tuple(segments)
        if (
            not self._segments_enabled
            or _segments_size(frozen_segments) > self._segments.maxsize
        ):
            return
        with self.lock:
            self._segments[key] = frozen_segments

    def clear(self) -> None:
        with self.lock:
            self._responses.clear()
            self._segments.clear()


_MEDIA_SOURCES_KEY = "eaf_base_api.load_sources"


class _UnloadedValue:
    """Private marker that distinguishes an unresolved field from a real ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNLOADED>"

    def __copy__(self) -> _UnloadedValue:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _UnloadedValue:
        return self

    def __reduce__(self) -> tuple[Callable[[], _UnloadedValue], tuple[()]]:
        return (_unloaded_value, ())


_UNLOADED = _UnloadedValue()


def _unloaded_value() -> _UnloadedValue:
    """Restore the process-wide sentinel while unpickling or deep-copying."""
    return _UNLOADED


def media_field(
    *sources: str, # Tuple that defines the sources in which the field can appear e.g., (html, api)
    default: Any = MISSING, # The default state for the field is missing cuz yeah it hasn't been loaded yet
    default_factory: Callable[[], Any] | Any = MISSING,
    repr: bool = False,
    compare: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """
    Declare a dataclass field that can be populated by one or more media loaders.

    Source order is significant: the first source has the highest precedence when
    several successfully loaded sources contain the same field.  If neither a
    ``default`` nor a ``default_factory`` is supplied, the field starts with an
    internal *unloaded* sentinel.  Consequently, a loader can return ``None`` and
    direct attribute access will correctly treat that value as loaded.

    ``repr`` and ``compare`` default to false because generated dataclass methods
    should not accidentally access an unresolved field.  Use ``BaseMedia.to_dict``
    when serialising a partially loaded model.

    Example::

        title: str | None = media_field("api", "html")
        stream_url: str | None = media_field("html")
    """
    if not sources:
        raise ValueError("media_field() requires at least one loader source")

    normalized_sources: list[str] = [] # Converts it into a list
    for source in sources:
        if not isinstance(source, str) or not source or not source.isidentifier(): # just checks if source is a valid string
            raise ValueError(
                "media loader sources must be non-empty Python identifiers; "
                f"received {source!r}"
            )
        if source in normalized_sources: # You shouldn't provide ("html", "html") because why?
            raise ValueError(f"media field source {source!r} was declared twice")
        normalized_sources.append(source)

    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("media_field() cannot receive both default and default_factory")

    field_metadata = dict(metadata or {})
    if _MEDIA_SOURCES_KEY in field_metadata: # You can't put the key for the field metadata as a field metadata because this interferes with the logic here, this should never happen though
        raise ValueError(f"metadata key {_MEDIA_SOURCES_KEY!r} is reserved")
    field_metadata[_MEDIA_SOURCES_KEY] = tuple(normalized_sources) # Creates the field metadata with the source key + tuple of normalized sources e.g., (html, api)

    field_arguments: dict[str, Any] = {
        "repr": repr,
        "compare": compare,
        "metadata": field_metadata,
    }

    # Repr and compare (__repr__ / __eq__) are False by default because if the element is not yet loaded
    # and you try to log it, this would cause an error
    # metadata holds the prepared dictionary of field / sources along with the media source key

    if default_factory is not MISSING:
        field_arguments["default_factory"] = default_factory
        # If we just give [] or {} as a default value in an object this will raise an issue cuz Python
        # for security reasons doesn't allow this because the memory would be shared between the different dataclass
        # instances. That's why we give a list, dict function which then creates the actual [] or {} when the dataclass
        # is created (yeah I know this seems weird, but needed)
    elif default is not MISSING:
        field_arguments["default"] = default
        # If a default value was provided the field will automatically get this value and it won't be tried to load
    else:
        field_arguments["default"] = _UNLOADED
        # We pick _UNLOADED here because the problem is that Python defaults to "None" for stuff that is
        # really not Available or is just Empty. The problem is, that I can't distinguish then if something
        # just isn't available because the HTML didn't expose it for example or if it's just empty.
        # With _UNLOADED we can definitely know for sure that the element is NOT yet available and NEEDS to be loaded

    return field(**field_arguments) # Creates the dataclass with the keyword values


class LoadState(StrEnum):
    """Observable lifecycle of one named ``BaseMedia`` source."""
    NOT_LOADED = "not_loaded" # (html or api or whatever was not yet loaded(
    LOADING = "loading" # It is currently loading e.g., being fetched
    LOADED = "loaded" # It finished loading (this is good), don't need to fetch it again :)
    FAILED = "failed" # It failed loading :(


@dataclass(frozen=True, slots=True)
class _MediaSchema:
    """Cached, validated view of a media dataclass's loadable fields."""
    field_names: frozenset[str] # Set of field names e.g., ('title', 'description')
    field_sources: dict[str, tuple[str, ...]] # Source map e.g.,: {'title': ('api', 'html')
    source_fields: dict[str, frozenset[str]] # like field_sources but in reverse
    source_order: tuple[str, ...] # Source order, usually ('api', 'html') cuz API is faster to load


_MEDIA_SCHEMA_CACHE: dict[type[Any], _MediaSchema] = {}
# Preserves a Cache of MediaScheme, because looking this up each time without would take time, this makes it faster


def _media_schema(model_type: type[Any]) -> _MediaSchema:
    """Build source/field indexes once per concrete dataclass type."""
    cached = _MEDIA_SCHEMA_CACHE.get(model_type)
    if cached is not None:
        return cached # Loads from cache

    # Model Type is needed to tell the base layout of the class.
    # If we do this per instance this would run as often as the instance appears which is unnecessary
    # Because why would I need to build the media scheme 5000 times when I can just do it once, tell the future
    # references which type it is and fetch it from cache, because guess what I won't randomly change
    # the scheme mid runtime xD
    dataclass_fields = tuple(fields(model_type)) # Variable annotations + metadata
    field_sources: dict[str, tuple[str, ...]] = {} # Creates a mutable dict out of the field sources
    source_fields_mutable: dict[str, set[str]] = {} # Creates a mutable dictionary out of the source fields
    source_order: list[str] = [] # See above
    # The fields need to be mutable here, because each new defined object in my classes need to be added
    # to the dictionary. So this is only temporary and down below this is going back to a frozenset

    for dataclass_field in dataclass_fields:
        # Get the tuple in the _MEDIA_SOURCES_KEY
        sources = dataclass_field.metadata.get(_MEDIA_SOURCES_KEY)
        # sources = e.g., ("api", "html")
        if not sources:
            continue # Regular dataclass fields are ignored e.g., 'url' and 'core' for most classes

        field_sources[dataclass_field.name] = tuple(sources)
        # Add e.g, title to field_sources, becomes {"title": ("api", "html")}
        for source in sources:
            # .setdefault("api", set()) Does e.g., 'api' exist in this dictionary. If not
            # Create it and return an empty set and add the dataclass field name
            # So it becomes {"api": {"title"}}
            source_fields_mutable.setdefault(source, set()).add(dataclass_field.name)
            if source not in source_order:
                source_order.append(source)

    schema = _MediaSchema(
        field_names=frozenset(item.name for item in dataclass_fields), # {"title", "duration", "raw_html"
        field_sources=field_sources, # {"title": ("api", "html")
        source_fields={
            source: frozenset(field_names) # "api": frozenset({"title", "duration"})
            for source, field_names in source_fields_mutable.items()
        },
        source_order=tuple(source_order), # ("api", "html")
    )
    _MEDIA_SCHEMA_CACHE[model_type] = schema # Loads it into the cache
    return schema # Return the final scheme


@dataclass(slots=True, kw_only=True, repr=False)
class BaseMedia:
    """
    Base class for dataclass models whose fields are loaded from remote sources.

    Subclasses declare loadable attributes with :func:`media_field` and map each
    source name to an async method through ``loader_methods``.  A loader returns a
    mapping; it never mutates the model itself.  ``BaseMedia`` validates the full
    mapping and commits it atomically, so a failed parser cannot leave half-loaded
    model state behind.

    Different callers requesting the same source share one task. Cancelling a
    waiter cancels that shared operation so network work never escapes its caller;
    every waiter then observes cancellation and the source becomes retryable.
    Loading different sources may happen concurrently, but field precedence stays
    deterministic because it follows the order declared by ``media_field``.
    """

    url: str
    core: object

    loader_methods: ClassVar[Mapping[str, str]] = {}

    _load_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )
    _source_states: dict[str, LoadState] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_results: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_errors: dict[str, BaseException] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _source_tasks: dict[str, asyncio.Task[None]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __getattribute__(self, name: str) -> Any:
        """
        Reject only the private unloaded sentinel, never a legitimate ``None``.

        Attribute access remains synchronous and therefore cannot initiate network
        I/O.  The exception tells callers exactly which field and sources to pass
        to ``load_fields`` or ``load_sources``.
        """
        value = object.__getattribute__(self, name)
        if value is not _UNLOADED:
            return value

        # Because we override the __getattribute__ method, we need to call object.__getattribute__.
        # If I'd use self.<something> we would get a recursion error as the function would call itself infinitely

        model_type = type(self)
        """
        Explanation why model_type exists:
        So, basically when we build the schema (down below) this takes some time. However, this function actually
        caches the fully resolved dataclass. So if we are going over a thousand Video objects usually we would need
        to reconstruct the thousand video objects, well, a thousand times.
        
        By giving the model_type with self, we can tell the cache: Yo, this is a Video object. See if you have
        already processed this and if yes it will use this instead of processing it each time again.
        Might only save a few milliseconds but I ain't Ubisoft XDDD
        """
        schema = _media_schema(model_type)
        sources = schema.field_sources.get(name, ()) # E.g., ("api", "html") for name=title (PornHub API as an example)
        url = object.__getattribute__(self, "url") # The actual URL to fetch e.g., https://example.com/video/id?=
        all_source_errors = object.__getattribute__(self, "_source_errors")
        relevant_errors = {
            source: all_source_errors[source]
            for source in sources
            if source in all_source_errors
        } # Basically just checks which sources had an error e.g., if html failed fetching this will return html
        raise DataNotLoadedError(
            model_type.__name__, name, url, sources, relevant_errors
            # Raises a custom exception that tells you which attribute failed with its associated source and the error
            # that happened along with the actual URL (so that I can replicate it when you report it)
        )

    def __repr__(self) -> str:
        """Represent identity and load state without touching unresolved fields."""
        loaded = ", ".join(sorted(self.loaded_sources)) or "none"
        return f"{type(self).__name__}(url={self.url!r}, loaded_sources={loaded})"

    @property
    def loaded_sources(self) -> frozenset[str]:
        """Return an immutable snapshot of sources that loaded successfully."""
        states = object.__getattribute__(self, "_source_states")
        return frozenset(
            source for source, state in states.items() if state is LoadState.LOADED
        ) # Returns the sources that have been successfully fetched

    @property
    def source_errors(self) -> Mapping[str, BaseException]:
        """Return a copy of the most recent failure for each source."""
        return dict(object.__getattribute__(self, "_source_errors"))

    def source_state(self, source: str) -> LoadState:
        """Inspect one declared source without starting a load."""
        schema = _media_schema(type(self))
        if source not in schema.source_fields:
            raise LoaderConfigurationError(
                f"{type(self).__name__} has no media fields assigned to source {source!r}"
            )
        return object.__getattribute__(self, "_source_states").get(
            source, LoadState.NOT_LOADED
        ) # Basically tells you the state of the source

    def is_field_loaded(self, field_name: str) -> bool:
        """Return whether a field contains a real value, including real ``None``."""
        self._validate_field_name(field_name)
        return object.__getattribute__(self, field_name) is not _UNLOADED
        # Basically checks if a field has been loaded

    def unloaded_fields(self) -> frozenset[str]:
        """Return all declared media fields that still contain the sentinel."""
        schema = _media_schema(type(self))
        return frozenset(
            field_name
            for field_name in schema.field_sources
            if object.__getattribute__(self, field_name) is _UNLOADED
        ) # Returns a set with all fields that have not yet been loaded

    def to_dict(
        self,
        *,
        include_unloaded: bool = False,
        include_core: bool = False,
    ) -> dict[str, Any]:
        """
        Serialise public dataclass fields without triggering lazy-field errors.

        Unresolved fields are omitted by default.  When ``include_unloaded`` is
        true they are represented as ``None``; this keeps the private sentinel out
        of application data.  ``core`` is excluded by default because clients and
        sessions are normally not serialisable.
        """
        result: dict[str, Any] = {}
        for dataclass_field in fields(type(self)):
            name = dataclass_field.name
            if name.startswith("_") or (name == "core" and not include_core):
                continue
            value = object.__getattribute__(self, name)
            if value is _UNLOADED:
                if include_unloaded:
                    result[name] = None
                continue
            result[name] = value
        return result

    async def load_sources(
        self,
        *sources: str,
        retry_failed: bool = True,
    ) -> Self:
        """
        Load named sources concurrently and return this model.

        Successful sources remain committed if a sibling source fails.  One
        failure is raised directly; several failures are wrapped in
        ``MediaLoadErrors`` rather than an ``ExceptionGroup``.
        """
        normalized_sources = tuple(dict.fromkeys(sources))
        if not normalized_sources:
            return self

        schema = _media_schema(type(self))
        for source in normalized_sources:
            if source not in schema.source_fields:
                raise LoaderConfigurationError(
                    f"{type(self).__name__} has no media fields assigned to "
                    f"source {source!r}"
                )
            self._loader_for_source(source)

        results = await asyncio.gather(
            *(
                self._ensure_source(source, retry_failed=retry_failed)
                for source in normalized_sources
            ),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                failures.append(result)

        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise MediaLoadErrors(tuple(failures))
        return self

    async def load_fields(
        self,
        *field_names: str,
        retry_failed: bool = True,
    ) -> Self:
        """
        Load the smallest useful set of sources for the requested fields.

        The source selection is a deterministic greedy cover.  For example, if
        ``html`` can populate both requested fields while ``api`` can populate
        only one, only ``html`` is loaded.  Ties follow the precedence order in
        the field declarations.
        """
        requested = tuple(dict.fromkeys(field_names))
        if not requested:
            return self

        schema = _media_schema(type(self))
        pending: set[str] = set()
        for field_name in requested:
            self._validate_field_name(field_name)
            if object.__getattribute__(self, field_name) is not _UNLOADED:
                continue
            if field_name not in schema.field_sources:
                raise FieldNotLoadableError(type(self).__name__, field_name)
            pending.add(field_name)

        selected_sources: list[str] = []
        while pending:
            best_source: str | None = None
            best_coverage: set[str] = set()
            best_preference_cost: int | None = None
            for source in schema.source_order:
                coverage = pending.intersection(schema.source_fields[source])
                preference_cost = sum(
                    schema.field_sources[field_name].index(source)
                    for field_name in coverage
                )
                if (
                    len(coverage) > len(best_coverage)
                    or (
                        len(coverage) == len(best_coverage)
                        and coverage
                        and (
                            best_preference_cost is None
                            or preference_cost < best_preference_cost
                        )
                    )
                ):
                    best_source = source
                    best_coverage = coverage
                    best_preference_cost = preference_cost

            if best_source is None:
                # Schema construction guarantees this cannot happen unless model
                # metadata was modified at runtime after it had been cached.
                unresolved = sorted(pending)
                raise LoaderConfigurationError(
                    f"No loader source can resolve fields {unresolved!r} on "
                    f"{type(self).__name__}"
                )
            selected_sources.append(best_source)
            pending.difference_update(best_coverage)

        return await self.load_sources(
            *selected_sources, retry_failed=retry_failed
        )

    async def get_field(self, field_name: str, *, retry_failed: bool = True) -> Any:
        """Load one field if necessary and return its value."""
        self._validate_field_name(field_name)
        if object.__getattribute__(self, field_name) is _UNLOADED:
            await self.load_fields(field_name, retry_failed=retry_failed)
        return object.__getattribute__(self, field_name)

    def _validate_field_name(self, field_name: str) -> None:
        schema = _media_schema(type(self))
        if field_name not in schema.field_names:
            raise UnknownMediaFieldError(type(self).__name__, field_name)

    def _loader_for_source(self, source: str) -> Callable[[], Awaitable[Mapping[str, Any]]]:
        method_name = type(self).loader_methods.get(source)
        if method_name is None:
            raise LoaderConfigurationError(
                f"{type(self).__name__}.loader_methods does not map source {source!r}"
            )
        try:
            loader = object.__getattribute__(self, method_name)
        except AttributeError as error:
            raise LoaderConfigurationError(
                f"{type(self).__name__}.loader_methods maps {source!r} to missing "
                f"method {method_name!r}"
            ) from error
        if not callable(loader):
            raise LoaderConfigurationError(
                f"{type(self).__name__}.{method_name} is not callable"
            )
        return cast(Callable[[], Awaitable[Mapping[str, Any]]], loader)

    async def _ensure_source(self, source: str, *, retry_failed: bool) -> None:
        """Return after one shared source task succeeds, or re-raise its error."""
        lock = object.__getattribute__(self, "_load_lock")
        async with lock:
            states = object.__getattribute__(self, "_source_states")
            tasks = object.__getattribute__(self, "_source_tasks")
            errors = object.__getattribute__(self, "_source_errors")
            state = states.get(source, LoadState.NOT_LOADED)

            if state is LoadState.LOADED:
                return
            if state is LoadState.FAILED and not retry_failed:
                raise errors[source]

            task = tasks.get(source)
            if task is None:
                states[source] = LoadState.LOADING
                task = asyncio.create_task(
                    self._execute_source_loader(source),
                    name=f"{type(self).__name__}:{source}:{self.url}",
                )
                tasks[source] = task

        # Direct awaiting intentionally propagates cancellation into the shared
        # source task. This keeps source I/O inside the lifetime of its callers.
        await task

    async def _execute_source_loader(self, source: str) -> None:
        """Execute, validate, and atomically commit one source loader."""
        model_name = type(self).__name__
        try:
            loader = self._loader_for_source(source)
            awaitable = loader()
            if not inspect.isawaitable(awaitable):
                raise LoaderContractError(
                    model_name,
                    source,
                    self.url,
                    "the configured loader must be async and return an awaitable",
                )
            raw_result = await awaitable
            result = self._validate_loader_result(source, raw_result)

            lock = object.__getattribute__(self, "_load_lock")
            async with lock:
                object.__getattribute__(self, "_source_results")[source] = result
                object.__getattribute__(self, "_source_states")[source] = LoadState.LOADED
                object.__getattribute__(self, "_source_errors").pop(source, None)
                self._apply_source_precedence(source)
                object.__getattribute__(self, "_source_tasks").pop(source, None)

        except asyncio.CancelledError:
            lock = object.__getattribute__(self, "_load_lock")
            async with lock:
                object.__getattribute__(self, "_source_states")[source] = LoadState.NOT_LOADED
                object.__getattribute__(self, "_source_tasks").pop(source, None)
            raise
        except Exception as error:
            recorded_error: BaseException
            if isinstance(error, (LoaderContractError, LoaderConfigurationError)):
                recorded_error = error
            else:
                recorded_error = MediaLoadError(
                    model_name, source, self.url, error
                )

            lock = object.__getattribute__(self, "_load_lock")
            async with lock:
                object.__getattribute__(self, "_source_states")[source] = LoadState.FAILED
                object.__getattribute__(self, "_source_errors")[source] = recorded_error
                object.__getattribute__(self, "_source_tasks").pop(source, None)

            if recorded_error is error:
                raise
            raise recorded_error from error

    def _validate_loader_result(
        self, source: str, raw_result: Any
    ) -> dict[str, Any]:
        """Enforce the all-fields, no-surprises loader mapping contract."""
        model_name = type(self).__name__
        if not isinstance(raw_result, Mapping):
            raise LoaderContractError(
                model_name,
                source,
                self.url,
                f"expected a mapping, received {type(raw_result).__name__}",
            )

        result = dict(raw_result)
        if not all(isinstance(name, str) for name in result):
            raise LoaderContractError(
                model_name, source, self.url, "all result keys must be strings"
            )
        if any(value is _UNLOADED for value in result.values()):
            raise LoaderContractError(
                model_name,
                source,
                self.url,
                "a loader may not return BaseMedia's private unloaded sentinel",
            )

        expected_fields = _media_schema(type(self)).source_fields[source]
        actual_fields = set(result)
        missing = sorted(expected_fields.difference(actual_fields))
        unexpected = sorted(actual_fields.difference(expected_fields))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing fields {missing!r}; return None when absent")
            if unexpected:
                details.append(f"unexpected fields {unexpected!r}")
            raise LoaderContractError(
                model_name, source, self.url, "; ".join(details)
            )
        return result

    def _apply_source_precedence(self, completed_source: str) -> None:
        """
        Recompute affected fields from loaded source snapshots.

        This method is called while ``_load_lock`` is held.  Looking through each
        field's sources in declaration order makes the final value independent of
        network completion order.
        """
        schema = _media_schema(type(self))
        states = object.__getattribute__(self, "_source_states")
        results = object.__getattribute__(self, "_source_results")
        for field_name in schema.source_fields[completed_source]:
            for candidate_source in schema.field_sources[field_name]:
                if states.get(candidate_source) is LoadState.LOADED:
                    object.__setattr__(
                        self,
                        field_name,
                        results[candidate_source][field_name],
                    )
                    break





MediaT = TypeVar("MediaT", bound=BaseMedia)
OperationT = TypeVar("OperationT")


@dataclass(frozen=True, slots=True)
class ScrapeResult(Generic[MediaT]):
    """
    Immutable result yielded for an item success or a configured stage failure.

    A successful result has ``item`` and no ``error``.  A yielded failure has an
    ``error`` and no ``item``.  Page successes are internal and are not yielded.
    """

    stage: ScrapeStage
    url: str
    page_index: int
    item_index: int | None
    attempts: int
    item: MediaT | None = None
    error: ScrapeOperationError | None = None

    def __post_init__(self) -> None:
        if (self.item is None) == (self.error is None):
            raise ValueError("ScrapeResult must contain exactly one of item or error")
        if self.stage is ScrapeStage.PAGE and self.item is not None:
            raise ValueError("a page-stage ScrapeResult cannot contain an item")

    @property
    def succeeded(self) -> bool:
        """Return true only for a successfully constructed and loaded item."""
        return self.error is None

    def unwrap(self) -> MediaT:
        """Return the item or raise the typed terminal scrape error."""
        if self.error is not None:
            raise self.error
        return cast(MediaT, self.item)


class ScrapeStream(Generic[MediaT]):
    """
    Async iterator/context manager owning a Helper scheduler.

    Exhausting the iterator cleans it up naturally.  When a caller may ``break``
    early, use ``async with`` so ``__aexit__`` immediately cancels outstanding page
    and item tasks instead of waiting for async-generator garbage collection.
    """

    def __init__(self, generator: AsyncGenerator[ScrapeResult[MediaT], None]) -> None:
        self._generator = generator
        self._closed = False

    def __aiter__(self) -> ScrapeStream[MediaT]:
        return self

    async def __anext__(self) -> ScrapeResult[MediaT]:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise

    async def __aenter__(self) -> ScrapeStream[MediaT]:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel scheduler work and close the underlying async generator once."""
        if self._closed:
            return
        self._closed = True
        await self._generator.aclose()


@dataclass(frozen=True, slots=True)
class _PageJob:
    index: int # The Index of the Page
    url: str # The URL of the Page


@dataclass(frozen=True, slots=True)
class _ItemJob:
    page_index: int # The Index of the Page (where the item was fetched from)
    item_index: int # The Item Index (needed to keep total order)
    url: str # The URL of the Item e.g., Video, Short URL
    data: dict[str, Any] # The actual item data defined by the extractor. Given into the constructor class


@dataclass(frozen=True, slots=True)
class _AttemptOutcome(Generic[OperationT]):
    value: OperationT | None
    error: ScrapeOperationError | None
    action: ErrorAction | None
    attempts: int


@dataclass(frozen=True, slots=True)
class _PageOutcome(Generic[MediaT]):
    job: _PageJob # Contains the Page Job class
    items: tuple[_ItemJob, ...] # Contains the Items, so the Videos, Shorts whatever (as ItemJob class)
    result: ScrapeResult[MediaT] | None # The actual Scrape Result


@dataclass(frozen=True, slots=True)
class _ItemOutcome(Generic[MediaT]):
    job: _ItemJob # The Item Job class
    result: ScrapeResult[MediaT] | None # The Scrape Result


class _OrderedResultBuffer(Generic[MediaT]):
    """Isolate original-order bookkeeping from the concurrency scheduler."""

    def __init__(self, page_count: int) -> None:
        self._page_count = page_count
        self._page_sizes: dict[int, int] = {}
        self._page_results: dict[int, ScrapeResult[MediaT] | None] = {}
        self._item_results: dict[
            tuple[int, int], ScrapeResult[MediaT] | None
        ] = {}
        self._next_page = 0
        self._next_item = 0
        self._page_result_emitted = False

    def add_page(self, outcome: _PageOutcome[MediaT]) -> None:
        self._page_sizes[outcome.job.index] = len(outcome.items)
        self._page_results[outcome.job.index] = outcome.result

    def add_item(self, outcome: _ItemOutcome[MediaT]) -> None:
        self._item_results[(outcome.job.page_index, outcome.job.item_index)] = (
            outcome.result
        )

    def drain(self) -> list[ScrapeResult[MediaT]]:
        """Return every now-contiguous result in page/extractor order."""
        ready: list[ScrapeResult[MediaT]] = []
        while self._next_page < self._page_count:
            if self._next_page not in self._page_sizes:
                break

            if not self._page_result_emitted:
                page_result = self._page_results.pop(self._next_page)
                self._page_result_emitted = True
                if page_result is not None:
                    ready.append(page_result)

            page_size = self._page_sizes[self._next_page]
            blocked = False
            while self._next_item < page_size:
                key = (self._next_page, self._next_item)
                if key not in self._item_results:
                    blocked = True
                    break
                item_result = self._item_results.pop(key)
                self._next_item += 1
                if item_result is not None:
                    ready.append(item_result)

            if blocked:
                break

            self._page_sizes.pop(self._next_page)
            self._next_page += 1
            self._next_item = 0
            self._page_result_emitted = False
        return ready


class Helper(Generic[MediaT]):
    """
    Concurrent two-stage scraper using bounded, dynamically managed task sets.

    Page tasks fetch and extract item dictionaries.  Item tasks construct a
    ``BaseMedia`` subclass and optionally load selected fields or sources.  There
    are no permanent workers, queues, sentinels, queue joins, or ``TaskGroup``.
    Completion is the explicit condition that page input, pending items, and both
    task sets are empty.

    ``ResultOrder.COMPLETION`` is the default and yields items as soon as their
    tasks finish. ``ResultOrder.ORIGINAL`` buffers only completed outcomes needed
    to restore target-page order and extractor order.
    """

    def __init__(
        self,
        core: BaseCore,
        constructor: Callable[..., MediaT],
        *,
        logger: logging.Logger | None = None,
        log_name: str = "helper.iterator",
        log_file: str | None = None,
        log_level: int = logging.INFO,
        http_ip: str | None = None,
        http_port: int | str | None = None,
    ) -> None:
        self.core = core # The Networking Backend
        self.constructor = constructor # The class that takes the data as input (dataclass) defined by each API
        self.logger = logger or configure_app_logging(
            log_name,
            log_file=log_file,
            level=log_level,
            http_ip=http_ip,
            http_port=http_port,
        )

    def iterator(
        self,
        target_page_urls: Sequence[str], # This is just a list of target URLs to scrape from
        item_extractor: Callable[[Any], Iterable[Mapping[str, Any]]], # The extractor that uses selectolax to parse data from the page
        *,
        iterator_config: IteratorConfig
    ) -> ScrapeStream[MediaT]:
        """
        Create a lazily started scrape stream.

        Extractors are synchronous callables returning an iterable of mappings.
        By default the complete extractor iteration runs in a worker thread so
        HTML parsing cannot block the event loop. Every mapping must contain a
        non-empty string under ``item_url_key`` and must be accepted as keyword
        arguments by ``constructor`` in addition to ``core``.

        ``load_sources`` runs before ``load_fields`` for each new instance. Both
        are optional; with neither configured, Helper only constructs models.
        Expected failures follow each stage's bounded retry policy and terminal
        error mode. Handler decisions can override the terminal mode but cannot
        exceed ``RetryPolicy.max_attempts``.
        """
        urls = tuple(target_page_urls)
        iterator_config = iterator_config.resolve(self.core.configuration)

        max_page_concurrency = iterator_config.max_page_concurrency
        max_item_concurrency = iterator_config.max_item_concurrency
        max_pending_items = iterator_config.max_pending_items
        extract_in_thread = iterator_config.extract_in_thread
        order = iterator_config.order
        page_error_mode = iterator_config.page_error_mode
        item_error_mode = iterator_config.item_error_mode
        page_retry = iterator_config.page_retry
        item_retry = iterator_config.item_retry
        page_error_handler = iterator_config.page_error_handler
        item_error_handler = iterator_config.page_error_handler
        load_fields = iterator_config.load_specific_fields
        load_sources = iterator_config.load_specific_sources
        page_request_method = iterator_config._page_request_method
        item_url_key = iterator_config._item_url_key

        # Validation of inputs
        if any(not isinstance(url, str) or not url for url in urls):
            raise ValueError("target_page_urls must contain non-empty strings")
        if not callable(item_extractor):
            raise TypeError("item_extractor must be callable")
        if max_page_concurrency < 1 or max_item_concurrency < 1:
            raise ValueError("page and item concurrency must both be at least 1")
        if max_pending_items is None:
            max_pending_items = max_item_concurrency * 4
        if max_pending_items < 1:
            raise ValueError("max_pending_items must be at least 1")
        if not isinstance(item_url_key, str) or not item_url_key:
            raise ValueError("item_url_key must be a non-empty string")

        normalized_order = ResultOrder(order)
        normalized_page_mode = ErrorMode(page_error_mode)
        normalized_item_mode = ErrorMode(item_error_mode)
        normalized_fields = tuple(dict.fromkeys(load_fields))
        normalized_sources = tuple(dict.fromkeys(load_sources))

        generator = self._iterate(
            urls=urls,
            item_extractor=item_extractor,
            max_page_concurrency=max_page_concurrency,
            max_item_concurrency=max_item_concurrency,
            max_pending_items=max_pending_items,
            page_request_method=page_request_method,
            item_url_key=item_url_key,
            extract_in_thread=extract_in_thread,
            load_fields=normalized_fields,
            load_sources=normalized_sources,
            order=normalized_order,
            page_error_mode=normalized_page_mode,
            item_error_mode=normalized_item_mode,
            page_retry=page_retry or RetryPolicy(),
            item_retry=item_retry or RetryPolicy(),
            page_error_handler=page_error_handler,
            item_error_handler=item_error_handler,
        )
        return ScrapeStream(generator)

    async def _iterate(
        self,
        *,
        urls: tuple[str, ...],
        item_extractor: Callable[[Any], Iterable[Mapping[str, Any]]],
        max_page_concurrency: int,
        max_item_concurrency: int,
        max_pending_items: int,
        page_request_method: str,
        item_url_key: str,
        extract_in_thread: bool,
        load_fields: tuple[str, ...],
        load_sources: tuple[str, ...],
        order: ResultOrder,
        page_error_mode: ErrorMode,
        item_error_mode: ErrorMode,
        page_retry: RetryPolicy,
        item_retry: RetryPolicy,
        page_error_handler: ErrorHandler | None,
        item_error_handler: ErrorHandler | None,
    ) -> AsyncGenerator[ScrapeResult[MediaT], None]:
        page_cursor = 0
        pending_items: deque[_ItemJob] = deque()
        page_tasks: dict[asyncio.Task[_PageOutcome[MediaT]], _PageJob] = {}
        item_tasks: dict[asyncio.Task[_ItemOutcome[MediaT]], _ItemJob] = {}
        ordered = _OrderedResultBuffer[MediaT](len(urls))

        try:
            while (
                page_cursor < len(urls) # As long as not all URLs have been processed
                or page_tasks # As long as page tasks still exist
                or pending_items # As long as pending items still exist (not yet fired up as a task)
                or item_tasks # As long as item tasks still exit
            ):
                # A bounded backlog provides backpressure: when item processing is
                # slower than page extraction, no additional pages are started.
                while (
                    page_cursor < len(urls) # As long as not all URLs have been processed
                    and len(page_tasks) < max_page_concurrency # As long as the page task count is smaller than the total page concurrency
                    and len(pending_items) < max_pending_items # __.__
                ):
                    page_job = _PageJob(page_cursor, urls[page_cursor]) # Creates a job for fetching a page
                    page_task = asyncio.create_task(
                        self._process_page(
                            page_job,
                            item_extractor=item_extractor,
                            request_method=page_request_method,
                            item_url_key=item_url_key,
                            extract_in_thread=extract_in_thread,
                            retry_policy=page_retry,
                            error_mode=page_error_mode,
                            error_handler=page_error_handler,
                        ),
                        name=f"scrape-page-{page_job.index}",
                    )
                    page_tasks[page_task] = page_job
                    page_cursor += 1

                while pending_items and len(item_tasks) < max_item_concurrency:
                    item_job = pending_items.popleft()
                    item_task = asyncio.create_task(
                        self._process_item(
                            item_job,
                            load_fields=load_fields,
                            load_sources=load_sources,
                            retry_policy=item_retry,
                            error_mode=item_error_mode,
                            error_handler=item_error_handler,
                        ),
                        name=(
                            f"scrape-item-{item_job.page_index}-"
                            f"{item_job.item_index}"
                        ),
                    )
                    item_tasks[item_task] = item_job

                active_tasks: set[asyncio.Task[Any]] = set(page_tasks)
                active_tasks.update(item_tasks)
                if not active_tasks:
                    # The loop condition says work remains, so reaching this branch
                    # would indicate a scheduler invariant bug rather than a remote
                    # scrape failure.
                    raise RuntimeError("Helper scheduler has pending work but no active tasks")

                done, _ = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                ready_results: list[ScrapeResult[MediaT]] = []

                for generic_task in done:
                    if generic_task in page_tasks:
                        page_task = cast(
                            asyncio.Task[_PageOutcome[MediaT]], generic_task
                        )
                        page_tasks.pop(page_task)
                        page_outcome = page_task.result()

                        if order is ResultOrder.ORIGINAL:
                            ordered.add_page(page_outcome)
                        elif page_outcome.result is not None:
                            ready_results.append(page_outcome.result)

                        pending_items.extend(page_outcome.items)
                    else:
                        item_task = cast(
                            asyncio.Task[_ItemOutcome[MediaT]], generic_task
                        )
                        item_tasks.pop(item_task)
                        item_outcome = item_task.result()

                        if order is ResultOrder.ORIGINAL:
                            ordered.add_item(item_outcome)
                        elif item_outcome.result is not None:
                            ready_results.append(item_outcome.result)

                if order is ResultOrder.ORIGINAL:
                    ready_results.extend(ordered.drain())

                for result in ready_results:
                    yield result
        finally:
            # This is the single owner of all scheduler tasks. It runs on normal
            # exhaustion, typed failure, caller cancellation, or ScrapeStream.close.
            remaining_tasks: list[asyncio.Task[Any]] = [*page_tasks, *item_tasks]
            for task in remaining_tasks:
                task.cancel()
            if remaining_tasks:
                await asyncio.gather(*remaining_tasks, return_exceptions=True)

    async def _process_page(
        self,
        job: _PageJob,
        *,
        item_extractor: Callable[[Any], Iterable[Mapping[str, Any]]],
        request_method: str,
        item_url_key: str,
        extract_in_thread: bool,
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
    ) -> _PageOutcome[MediaT]:
        async def operation() -> tuple[_ItemJob, ...]:
            self.logger.debug("Fetching page %s: %s", job.index, job.url)
            content = await self.core.fetch_text(job.url, method=request_method)

            def extract_all() -> tuple[Mapping[str, Any], ...]:
                extracted = item_extractor(content)
                if inspect.isawaitable(extracted):
                    raise TypeError(
                        "item_extractor must be synchronous; Helper can move it "
                        "to a worker thread"
                    )
                return tuple(extracted)

            if extract_in_thread:
                extracted_items = await asyncio.to_thread(extract_all)
            else:
                extracted_items = extract_all()

            jobs: list[_ItemJob] = []
            for item_index, raw_item in enumerate(extracted_items):
                if not isinstance(raw_item, Mapping):
                    raise TypeError(
                        "item_extractor entries must be mappings; received "
                        f"{type(raw_item).__name__} at index {item_index}"
                    )
                data = dict(raw_item)
                item_url = data.get(item_url_key)
                if not isinstance(item_url, str) or not item_url:
                    raise ValueError(
                        f"extractor item {item_index} must contain a non-empty "
                        f"string under key {item_url_key!r}"
                    )
                jobs.append(_ItemJob(job.index, item_index, item_url, data))
            return tuple(jobs)

        attempt = await self._run_operation(
            operation,
            stage=ScrapeStage.PAGE,
            url=job.url,
            page_index=job.index,
            item_index=None,
            retry_policy=retry_policy,
            error_mode=error_mode,
            error_handler=error_handler,
            error_factory=lambda error, number: PageFetchError(
                job.url, error, number, job.index
            ),
        )
        if attempt.error is None:
            return _PageOutcome(job, cast(tuple[_ItemJob, ...], attempt.value), None)

        result: ScrapeResult[MediaT] | None = None
        if attempt.action is ErrorAction.YIELD:
            result = ScrapeResult(
                stage=ScrapeStage.PAGE,
                url=job.url,
                page_index=job.index,
                item_index=None,
                attempts=attempt.attempts,
                error=attempt.error,
            )
        return _PageOutcome(job, (), result)

    async def _process_item(
        self,
        job: _ItemJob,
        *,
        load_fields: tuple[str, ...],
        load_sources: tuple[str, ...],
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
    ) -> _ItemOutcome[MediaT]:
        async def operation() -> MediaT:
            instance = self.constructor(core=self.core, **job.data)
            if not isinstance(instance, BaseMedia):
                raise TypeError(
                    "Helper constructors must return a BaseMedia instance; "
                    f"received {type(instance).__name__}"
                )
            if load_sources:
                await instance.load_sources(*load_sources)
            if load_fields:
                await instance.load_fields(*load_fields)
            return instance

        attempt = await self._run_operation(
            operation,
            stage=ScrapeStage.ITEM,
            url=job.url,
            page_index=job.page_index,
            item_index=job.item_index,
            retry_policy=retry_policy,
            error_mode=error_mode,
            error_handler=error_handler,
            error_factory=lambda error, number: ItemFetchError(
                job.url,
                error,
                number,
                job.page_index,
                job.item_index,
            ),
        )
        if attempt.error is None:
            success_result = ScrapeResult(
                stage=ScrapeStage.ITEM,
                url=job.url,
                page_index=job.page_index,
                item_index=job.item_index,
                attempts=attempt.attempts,
                item=cast(MediaT, attempt.value),
            )
            return _ItemOutcome(job, success_result)

        failure_result: ScrapeResult[MediaT] | None = None
        if attempt.action is ErrorAction.YIELD:
            failure_result = ScrapeResult(
                stage=ScrapeStage.ITEM,
                url=job.url,
                page_index=job.page_index,
                item_index=job.item_index,
                attempts=attempt.attempts,
                error=attempt.error,
            )
        return _ItemOutcome(job, failure_result)

    async def _run_operation(
        self,
        operation: Callable[[], Awaitable[OperationT]],
        *,
        stage: ScrapeStage,
        url: str,
        page_index: int,
        item_index: int | None,
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
        error_factory: Callable[[Exception, int], ScrapeOperationError],
    ) -> _AttemptOutcome[OperationT]:
        """Run one bounded retry loop and convert its terminal disposition."""
        for attempt in range(1, retry_policy.max_attempts + 1):
            try:
                value = await operation()
                return _AttemptOutcome(value, None, None, attempt)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                context = ScrapeErrorContext(
                    stage=stage,
                    url=url,
                    error=error,
                    attempt=attempt,
                    max_attempts=retry_policy.max_attempts,
                    page_index=page_index,
                    item_index=item_index,
                )
                action = await self._error_action(
                    context,
                    retry_policy=retry_policy,
                    error_mode=error_mode,
                    error_handler=error_handler,
                )

                if action is ErrorAction.RETRY and attempt < retry_policy.max_attempts:
                    delay = retry_policy.delay_after(attempt)
                    self.logger.warning(
                        "Retrying %s %s after attempt %s/%s in %.3fs: %s",
                        stage.value,
                        url,
                        attempt,
                        retry_policy.max_attempts,
                        delay,
                        error,
                    )
                    if delay:
                        await asyncio.sleep(delay)
                    continue

                # A handler cannot create an unbounded retry loop. RETRY on the
                # final permitted attempt falls back to the configured mode.
                if action is ErrorAction.RETRY:
                    action = ErrorAction(error_mode.value)

                wrapped_error = error_factory(error, attempt)
                if action is ErrorAction.RAISE:
                    raise wrapped_error from error
                return _AttemptOutcome(None, wrapped_error, action, attempt)

        raise RuntimeError("retry loop exhausted without returning an outcome")

    async def _error_action(
        self,
        context: ScrapeErrorContext,
        *,
        retry_policy: RetryPolicy,
        error_mode: ErrorMode,
        error_handler: ErrorHandler | None,
    ) -> ErrorAction:
        """Resolve automatic policy or validate a custom handler decision."""
        if error_handler is None:
            if (
                context.attempt < retry_policy.max_attempts
                and retry_policy.permits(context.error)
            ):
                return ErrorAction.RETRY
            return ErrorAction(error_mode.value)

        try:
            decision = error_handler(context)
            if inspect.isawaitable(decision):
                decision = await decision
            if not isinstance(decision, ErrorAction):
                raise TypeError(
                    "error handlers must return an ErrorAction value, received "
                    f"{decision!r}"
                )
            return decision
        except asyncio.CancelledError:
            raise
        except Exception as handler_error:
            raise ErrorHandlerError(
                context.stage.value, context.url, handler_error
            ) from handler_error


class BaseCore:
    """
    The base class which has all necessary functions for other API packages
    """
    def __init__(
        self,
        configuration: "RuntimeConfig" = config,
        *,
        cache: CacheBackend | None = None,
    ) -> None:
        self.lock = asyncio.Lock()
        self._delay_lock = asyncio.Lock()
        self._cache_flight_lock = asyncio.Lock()
        self._inflight_text_requests: dict[RequestCacheKey, asyncio.Future[str]] = {}
        self.latest_key: str | None = None
        self.latest_key_time: float = 0.0
        self.last_request_time: float | None = None
        self.total_requests: int = 0  # Tracks how many requests have been made
        self.session: AsyncSession | None = None
        self.configuration = configuration
        self.cache = cache if cache is not None else Cache(self.configuration)
        self.logger = configure_app_logging("BASE API - [BaseCore]", log_file=None, level=logging.ERROR)
        self.default_headers = {
            "User-Agent": UA_DESKTOP_CHROME,
            "Accept-Language": self.configuration.locale,
            "Accept-Encoding": "gzip, deflate, br"
        }

    async def __aenter__(self) -> Self:
        if self.session is None:
            self.initialize_session()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the owned HTTP session and allow the core to be reused later."""
        session, self.session = self.session, None
        if session is not None:
            await session.close()

    def enable_logging(self, log_file: str | None = None, level: int = logging.DEBUG, log_ip:
    str | None = None, log_port: int | str | None = None) -> None:
        """Enables logging dynamically for this module."""
        self.logger = configure_app_logging("BASE API - [BaseCore]", log_file=log_file, level=level, http_ip=log_ip,
                                   http_port=log_port)

    def initialize_session(self) -> None:
        verify = self.configuration.verify_ssl

        curl_options: Dict[CurlOpt, Union[bytes, int]] = {}
        if self.configuration.dns_over_https:
            curl_options[CurlOpt.DOH_URL] = str(self.configuration.dns_over_https).encode("utf-8")

        proxy = None
        if self.configuration.proxy:
            proxy = self.configuration.proxy

        if self.configuration.max_bandwidth_mb is not None and self.configuration.max_bandwidth_mb > 0:
            global_limit_bytes = int(self.configuration.max_bandwidth_mb * 1024 * 1024)
            total_concurrent_connections = (self.configuration.max_workers_download *
                                            self.configuration.videos_concurrency)
            per_connection_limit = max(1, int(global_limit_bytes / total_concurrent_connections))
            curl_options[CurlOpt.MAX_RECV_SPEED_LARGE] = per_connection_limit

        js3 = self.configuration.custom_ja3
        impersonation = self.configuration.impersonation
        http_version = self.configuration.http_version
        proxy_auth_str = self.configuration.proxy_auth
        trust_env = self.configuration.trust_env

        p_auth: Tuple[str, str] | None = None
        if proxy_auth_str and ":" in proxy_auth_str:
            u, p = proxy_auth_str.split(":", 1)
            p_auth = (u, p)

        self.session = cast(Any, AsyncSession)(
            interface=self.configuration.interface,
            proxy=proxy,
            timeout=self.configuration.timeout,
            verify=verify,
            impersonate=impersonation,
            curl_options=curl_options,
            http_version=http_version,
            ja3=js3,
            proxy_auth=p_auth,
            trust_env=trust_env
        )
        # Ensure our defaults are on the session
        assert self.session is not None
        self.session.headers.update(self.default_headers)

    async def enforce_delay(self) -> None:
        """Enforces the specified delay in config.request_delay (only if > 0)."""
        delay = self.configuration.request_delay
        if delay and delay > 0:
            async with self._delay_lock:
                now = time.monotonic()
                if self.last_request_time is None:
                    self.last_request_time = now
                    return
                time_since_last_request = now - self.last_request_time
                self.logger.debug(
                    "Time since last request: %.2f seconds.", time_since_last_request
                )
                if time_since_last_request < delay:
                    sleep_time = delay - time_since_last_request
                    self.logger.debug("Enforcing delay of %.2f seconds.", sleep_time)
                    await asyncio.sleep(sleep_time)
                self.last_request_time = time.monotonic()

    def _merged_headers(self, override: Dict[str, str] | None) -> Dict[str, Any]:
        """
        Create request headers from current session headers + optional overrides.
        Overrides win, session headers are the base.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        headers: Dict[str, Any] = cast(Dict[str, Any], cast(Any, dict(session.headers)))
        if override:
            headers.update(override)
        return headers

    def _merged_cookies(self, override: Dict[str, str] | None) -> Dict[str, Any]:
        """Same as above, but for cookies"""
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None
        cookies: Dict[str, Any] = cast(Dict[str, Any], cast(Any, session.cookies.get_dict()))
        if override:
            cookies.update(override)
        return cookies

    async def request(
        self,
        url: str,
        *,
        timeout: float | None = None,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        retry_non_idempotent: bool = False,
    ) -> Response:
        """
        Execute an HTTP request and return a successful response.

        Network failures, HTTP 408/425/429, and 5xx responses are retried for
        idempotent methods. Retrying a non-idempotent method requires an explicit
        opt-in because the server may already have applied the request.
        """
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        request_method = method.upper()
        req_timeout = timeout if timeout is not None else self.configuration.timeout
        max_attempts = max(1, int(self.configuration.request_attempts))
        method_is_retryable = request_method in {
            "GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"
        } or retry_non_idempotent

        def should_retry(error: BaseException) -> bool:
            if not method_is_retryable:
                return False
            if isinstance(error, (RequestsError, NetworkRequestError)):
                return True
            return isinstance(error, HTTPStatusError) and (
                error.status_code in {408, 425, 429}
                or 500 <= error.status_code < 600
            )

        exponential_wait = wait_exponential_jitter(
            initial=self.configuration.request_retry_initial_delay,
            max=self.configuration.request_retry_max_delay,
            jitter=self.configuration.request_retry_jitter,
        )

        def retry_wait(retry_state: Any) -> float:
            error = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(error, RateLimitError) and error.retry_after is not None:
                return max(0.0, error.retry_after)
            return cast(float, exponential_wait(retry_state))

        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=retry_wait,
            retry=retry_if_exception(should_retry),
            reraise=False,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    try:
                        await self.enforce_delay()
                        req_headers = self._merged_headers(headers)
                        req_cookies = self._merged_cookies(cookies)
                        
                        current_time = asyncio.get_running_loop().time()
                        latest_key = self.latest_key
                        if "KEY" not in session.cookies and latest_key is not None:
                            if current_time - self.latest_key_time < 10:
                                session.cookies.set("KEY", latest_key, domain=".pornhub.com", path="/")

                        self.total_requests += 1
                        response = await cast(Any, session).request(
                            method=cast(Any, request_method),
                            url=url,
                            timeout=req_timeout,
                            allow_redirects=allow_redirects,
                            data=data,
                            json=json_data,
                            params=params,
                            headers=req_headers,
                            cookies=req_cookies,
                        )

                        status = response.status_code

                        content_type = response.headers.get("content-type", "").lower()
                        is_html = "text/html" in content_type if content_type else True

                        if is_html:
                            enc = getattr(response, "encoding", None) or "utf-8"
                            resp_text = cast(bytes, response.content).decode(enc, errors="replace")

                            if 'onload="go()"' in resp_text:
                                local_latest = getattr(self, "latest_key", None)
                                async with self.lock:
                                    if getattr(self, "latest_key", None) != local_latest:
                                        self.logger.info("Another task already resolved the challenge! Retrying request with the new cookie.")
                                        if self.latest_key:
                                            session.cookies.set("KEY", self.latest_key, domain=".pornhub.com", path="/")

                                        await asyncio.sleep(1.5)
                                        raise NetworkRequestError("Retrying request with the new cookie.")

                                    self.logger.info("Challenge page detected! Solving...")
                                    get_challenge = re.compile(r'go\(\).*?{(.*?)n=l.*?KEY.*?s\+":(\d+):', re.DOTALL)
                                    challenge_data = re.search(get_challenge, resp_text)

                                    if challenge_data:
                                        try:
                                            challenge_str, token_str = challenge_data.groups()
                                            code = parse_challenge(challenge_str)
                                            code = other_challenge(code)
                                            code = '\n'.join(code.split(';'))

                                            safe_chars = set(string.ascii_letters + string.digits + " \t\n=+-*/().:><&|~^")
                                            if not all(c in safe_chars for c in code):
                                                self.logger.error("Security Abort: Illegal chars in challenge, CODE: %s", code)
                                                raise SecurityAbort

                                            safe_globals: Dict[str, Any] = {"__builtins__": {}}
                                            safe_locals = {"p": 0, "s": 0}
                                            exec(code, safe_globals, safe_locals)

                                            p = safe_locals.get('p', 0)
                                            s = safe_locals.get('s', 0)
                                            n = least_factors(p)
                                            cookie_value = f'{n}*{p // n}:{s}:{token_str}:1'

                                            self.latest_key = cookie_value
                                            self.latest_key_time = asyncio.get_running_loop().time()
                                            session.cookies.set("KEY", cookie_value, domain=".pornhub.com", path="/")
                                            self.logger.info("RESOLVED CHALLENGE! Injected cookie: %s", cookie_value)

                                            try:
                                                self.cache.invalidate_url(url)
                                            except (KeyError, Exception):
                                                pass

                                            await asyncio.sleep(1.5)
                                            raise NetworkRequestError("Retrying request after solving challenge.")
                                        except (NetworkRequestError, SecurityAbort):
                                            raise
                                        except Exception as challenge_error:
                                            raise ChallengeMathError from challenge_error

                                    else:
                                        self.logger.error("Detected challenge page, but the regex failed to extract data.")
                                        await asyncio.sleep(1.5)
                                        raise ChallengeRegexError("Detected Challenge, but regex couldn't extract, report this!")


                        if 200 <= status < 300:
                            self.logger.debug("Successfully fetched URL: %s", url)
                            return response

                        if status in {401, 403}:
                            raise AccessDeniedError("Request blocked by server!")

                        if status == 412:
                            log_precondition_failed(logger=self.logger, attempt=attempt.retry_state.attempt_number, response=response)

                        if status == 410:
                            raise ResourceGone(f"Resource gone (HTTP 410) for URL: {url}")

                        if status == 429:
                            retry_after = parse_retry_after(
                                logger=self.logger, response=response
                            )
                            if retry_after is not None:
                                self.logger.warning(
                                    "Rate limited (429). Server requested %ss pause.",
                                    retry_after,
                                )
                            raise RateLimitError(
                                "429 Rate Limited", retry_after=retry_after, url=url
                            )

                        if 500 <= status < 600:
                            self.logger.warning("Server error %s on %s. Retrying...", status, url)
                            raise HTTPStatusError(f"Server error {status}", status_code=status, url=url)

                        self.logger.info("HTTP %s for %s.", status, url)
                        raise HTTPStatusError(
                            f"HTTP {status} for {url}", status_code=status, url=url
                        )

                    except RequestsError as e:
                        err_str = str(e).lower()
                        self.logger.error("Request error for URL %s: %s", url, e, exc_info=True)
                        if "certificate verify failed" in err_str:
                            raise ProxySSLError("Proxy has an invalid SSL certificate, set 'verify = False' in config") from e
                        elif "cookie conflict" in err_str:
                            raise UnknownError(f"Cookie conflict during request to {url}: {e}") from e
                        elif "proxy" in err_str:
                            raise InvalidProxy("Proxy error when trying a request, aborting!") from e
                        elif "timeout" in err_str or "read" in err_str:
                            self.logger.error("Timeout for URL %s: %s", url, e, exc_info=True)
                        raise
                    except (BaseScraperError, ResourceGone, ProxySSLError, InvalidProxy, UnknownError):
                        raise

                    except Exception as e:
                        self.logger.error("Unexpected error for %s: %s\n%s", url, e, traceback.format_exc())
                        raise UnknownError(f"Unexpected error for URL {url}: {e}") from e

        except RetryError as re_err:
            last_error = re_err.last_attempt.exception()
            if not isinstance(last_error, Exception):
                last_error = NetworkRequestError("Request retry budget was exhausted")
            self.logger.error(
                "Request to %s failed after %s attempts.", url, max_attempts
            )
            raise RequestRetriesExhausted(url, max_attempts, last_error) from last_error

        raise RuntimeError("request retry controller exited without an outcome")

    def _request_cache_key(
        self,
        *,
        url: str,
        method: str,
        allow_redirects: bool,
        params: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
        json_data: Mapping[str, Any] | None,
        headers: Dict[str, str] | None,
        cookies: Dict[str, str] | None,
    ) -> RequestCacheKey:
        merged_headers = {
            str(key).lower(): value
            for key, value in self._merged_headers(headers).items()
        }
        merged_cookies = self._merged_cookies(cookies)
        return RequestCacheKey(
            method=method.upper(),
            url=url,
            allow_redirects=allow_redirects,
            params_fingerprint=_cache_fingerprint(params),
            body_fingerprint=_cache_fingerprint((data, json_data)),
            headers_fingerprint=_cache_fingerprint(merged_headers),
            cookies_fingerprint=_cache_fingerprint(merged_cookies),
        )

    @staticmethod
    def _decode_response(response: Response, url: str, logger: logging.Logger) -> str:
        raw_content = cast(bytes, response.content)
        encoding = getattr(response, "encoding", None) or "utf-8"
        try:
            return raw_content.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            logger.warning(
                "Content could not be decoded as %s (%s), decoding latin1 instead!",
                encoding,
                url,
            )
            return raw_content.decode("latin1", errors="replace")

    async def fetch_text(
        self,
        url: str,
        *,
        cache_policy: CachePolicy = CachePolicy.USE,
        timeout: float | None = None,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        retry_non_idempotent: bool = False,
    ) -> str:
        """Fetch and decode text, optionally using the bounded response cache."""
        request_method = method.upper()
        cacheable = request_method == "GET" and cache_policy is not CachePolicy.BYPASS
        key = None
        if cacheable:
            key = self._request_cache_key(
                url=url,
                method=request_method,
                allow_redirects=allow_redirects,
                params=params,
                data=data,
                json_data=json_data,
                headers=headers,
                cookies=cookies,
            )

        if key is not None and cache_policy is CachePolicy.USE:
            cached = self.cache.get_response(key)
            if cached is not None:
                self.logger.info("Fetched content for %s from cache.", url)
                return cached

        leader = True
        pending: asyncio.Future[str] | None = None
        if key is not None:
            async with self._cache_flight_lock:
                if cache_policy is CachePolicy.USE:
                    cached = self.cache.get_response(key)
                    if cached is not None:
                        return cached
                pending = self._inflight_text_requests.get(key)
                if pending is None:
                    pending = asyncio.get_running_loop().create_future()
                    self._inflight_text_requests[key] = pending
                else:
                    leader = False

        if not leader:
            assert pending is not None
            return await asyncio.shield(pending)

        try:
            response = await self.request(
                url,
                timeout=timeout,
                cookies=cookies,
                allow_redirects=allow_redirects,
                data=data,
                method=request_method,
                headers=headers,
                json_data=json_data,
                params=params,
                retry_non_idempotent=retry_non_idempotent,
            )
            content = self._decode_response(response, url, self.logger)
            if key is not None:
                self.cache.set_response(key, content)
                assert pending is not None
                if not pending.done():
                    pending.set_result(content)
            return content
        except BaseException as error:
            if pending is not None and not pending.done():
                pending.set_exception(error)
                # Mark the exception as observed even when there were no followers.
                pending.exception()
            raise
        finally:
            if key is not None:
                async with self._cache_flight_lock:
                    if self._inflight_text_requests.get(key) is pending:
                        self._inflight_text_requests.pop(key, None)

    async def fetch_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        cookies: Dict[str, str] | None = None,
        allow_redirects: bool = True,
        data: Dict[str, Any] | None = None,
        method: str = "GET",
        headers: Dict[str, str] | None = None,
        json_data: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
        retry_non_idempotent: bool = False,
    ) -> bytes:
        """Fetch a response body as bytes without involving the text cache."""
        response = await self.request(
            url,
            timeout=timeout,
            cookies=cookies,
            allow_redirects=allow_redirects,
            data=data,
            method=method,
            headers=headers,
            json_data=json_data,
            params=params,
            retry_non_idempotent=retry_non_idempotent,
        )
        return cast(bytes, response.content)


    @lru_cache(maxsize=250)
    async def get_m3u8_by_quality(self, m3u8_url: str, quality: Union[str, int]) -> str:
        """
        Return the media-playlist URL for the requested quality.

        quality:
          - 'best' | 'half' | 'worst'
          - 1080 / '1080' / '1080p' (and similar)
        """
        if m3u8 is None:
            raise ModuleNotFoundError(f"""
Using m3u8 is optional depending whether you use HLS videos or static videos. It seems like you are trying to download
from HLS. Please install m3u8 using: `pip install m3u8`.

If this does not fix the issue, there's an import error related to your environment. In this case please create
a new Python file, import only m3u8 and see what error you get. 
""")

        # Resolve master content
        assert m3u8 is not None

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if m3u8_url.lstrip().startswith("#EXTM3U"):
            master = m3u8.loads(m3u8_url)
            self.logger.debug("Resolved inline/custom m3u8 master content.")
            base_for_join = ""  # URIs should be absolute in inline cases; join will handle if relative
        else:
            content = await self.fetch_text(url=m3u8_url)
            master = m3u8.loads(content)
            base_for_join = m3u8_url
            self.logger.debug("Resolved m3u8 master: %s", m3u8_url)

        if not master.is_variant:
            raise PlaylistExtractionError(f"Playlist is not a master Playlist: {m3u8_url}")

        variants = collect_variants(master)
        if not variants:
            raise PlaylistExtractionError(f"No usable variants found in master Playlist: {m3u8_url}, {master}")

        q = normalize_quality_value(quality)
        if isinstance(q, str):  # 'best'/'half'/'worst'
            chosen = pick_by_label(variants, q)
        else:  # numeric height like 1080, 720, etc.
            chosen = pick_by_height(variants, q)

        full_url = urljoin(base_for_join or m3u8_url, chosen["uri"])
        return full_url

    async def list_available_qualities(self, m3u8_url: str) -> List[int]:
        """
        Inspect the master playlist and return sorted unique heights (e.g., [240, 360, 480, 720, 1080]).
        """
        assert m3u8 is not None

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if not m3u8_url.startswith("https://"):
            master = m3u8.loads(m3u8_url)
        else:
            content = await self.fetch_text(url=m3u8_url)
            master = m3u8.loads(content)

        if not master.is_variant:
            return []

        heights = {h for h in (height_from_variant(v) for v in master.playlists) if h is not None}
        if heights:
            return sorted(heights)
        # fallback: bandwidth-only (roughly infer tiers)
        by_bw = sorted(
            (getattr(v.stream_info, "bandwidth", 0) for v in master.playlists if is_video_playlist(v)),
            key=int
        )
        # Return rank numbers instead of heights if we truly can't infer—kept simple:
        return [i for i, _ in enumerate(by_bw, start=1)]

    async def get_segments(self, m3u8_url_master: str, quality: Union[str, int]) -> List[str]:
        assert m3u8 is not None
        segment_cache_key = SegmentCacheKey(m3u8_url_master, str(quality))
        _segments = self.cache.get_segments(segment_cache_key)
        if _segments is not None:
            self.logger.info("Received: %s from cache!", len(_segments))
            return _segments

        # Resolve the quality-specific playlist URL (may still be a master in some edge cases)
        playlist_url = await self.get_m3u8_by_quality(m3u8_url=m3u8_url_master, quality=quality)
        self.logger.debug("Trying to fetch segments from m3u8 -> %s", playlist_url)

        # M3U8s are volatile → don't cache
        content = await self.fetch_text(
            url=playlist_url, cache_policy=CachePolicy.BYPASS
        )
        parsed = m3u8.loads(content)

        # If we accidentally got a master, pick the first media playlist (existing behavior),
        # and IMPORTANT: update base_url for urljoin to the *new* playlist URL.
        base_url = playlist_url
        if parsed.is_variant:
            self.logger.warning("Media playlist expected; got variant. Resolving to first sub-playlist...")
            media_rel = parsed.playlists[0].uri
            media_url = urljoin(playlist_url, media_rel)
            self.logger.info("Resolved to new URL: %s", media_url)
            content = await self.fetch_text(
                url=media_url, cache_policy=CachePolicy.BYPASS
            )
            parsed = m3u8.loads(content)
            base_url = media_url

        segments: List[str] = []

        # Robust init segment handling (EXT-X-MAP)
        # Older m3u8 lib: .segment_map; newer: .init_section
        init_url = None
        segments_map = getattr(parsed, "segment_map", None)
        if segments_map:
            assert isinstance(segments_map, list)
            try:
                init_url = urljoin(base_url, segments_map[0].uri)
            except Exception as exc:
                self.logger.info("Couldn't get init url, this is probably not an issue: %s", exc)
                pass
        if init_url is None:
            init_section = getattr(parsed, "init_section", None)
            if init_section and getattr(init_section, "uri", None):
                init_url = urljoin(base_url, init_section.uri)

        if init_url:
            segments.append(init_url)
            self.logger.debug("Found init segment: %s", init_url)

        # Build absolute URLs for all media segments
        for seg in parsed.segments:
            segments.append(urljoin(base_url, seg.uri))

        self.logger.debug("Fetched %s segments from m3u8 URL (including init if present)", len(segments))
        self.logger.info("Saving segments to cache....")
        self.cache.set_segments(segment_cache_key, segments)
        return segments


    def _safe_remove(self, path: str | None) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug("Failed to remove file %s: %s", path, e)

    def _safe_rmtree(self, path: str | None) -> None:
        if not path:
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.debug("Failed to remove directory %s: %s", path, e)

    async def download_segment(self, url: str, timeout: int, stop_event:
                                threading.Event | None = None) -> tuple[str, bytes, bool]:
        """
        Attempt to download a single segment.
        Returns (url, content, success).
        """
        try:
            if stop_event is not None and stop_event.is_set():
                return url, b"", False # Stopping the download here

            content = await self.fetch_bytes(url, timeout=timeout)
            return url, content, True
        except Exception as e:
            # Log and mark failure; the caller will decide whether to retry or abort.
            self.logger.warning("Segment download failed: %s -> %s", url, e)
            return url, b"", False

    async def download(
        self,
        configuration: DownloadConfigHLS
    ) -> DownloadReport | bool:
        """
        :param video:
        :param configuration:
        :return:
        """

        if configuration.callback is None:
            # Use a terminal text progressbar by default
            configuration.callback = Callback.text_progress_bar
            self.logger.debug("download: no callback provided, using default text progress bar")

        m3u8_url = configuration.m3u8_base_url

        if inspect.iscoroutinefunction(m3u8_url) or (callable(m3u8_url) and not isinstance(m3u8_url, str)):
            m3u8_url = m3u8_url()
        if inspect.iscoroutine(m3u8_url) or inspect.isawaitable(m3u8_url):
            m3u8_url = await m3u8_url

        if m3u8_url:
            self.logger.debug("Download m3u8_base_url=%s", m3u8_url)

        self.logger.debug("download: dispatching to threaded downloader (timeout=%s)", self.configuration.timeout)

        # 2. Call the downloader method directly
        return await self.threaded_download(
            configuration=configuration,
            pre_resolved_m3u8=m3u8_url,
            timeout=config.timeout,
            max_workers=config.max_workers_download
        )

    async def threaded_download(
        self: "BaseCore",
        timeout: int,
        max_workers: int,
        pre_resolved_m3u8: str,
        configuration: DownloadConfigHLS,
    ) -> DownloadReport | bool:
        """
        Threaded HLS segment downloader with optional resume state and stop flag.
        """
        try:
            cleanup_on_stop = configuration.cleanup_on_stop
            keep_segment_dir = configuration.keep_segment_dir
            quality = configuration.quality
            path = configuration.path
            remux = configuration.remux
            start_segment = configuration.start_segment
            segment_state_path = configuration.segment_state_path
            segment_dir = configuration.segment_dir
            return_report = configuration.return_report
            callback = configuration.callback
            callback_remux = configuration.callback_remux
            stop_event = configuration.stop_event
            ios_support = configuration.ios_support
            timeout = timeout
            pre_resolved_m3u8_url = pre_resolved_m3u8

            self.logger.info(
                f"Threaded download start: quality={quality} path={path} remux={remux} start_segment={start_segment} "
                f"segment_state_path={segment_state_path} segment_dir={segment_dir} return_report={return_report} "
                f"cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir} max_workers={max_workers} "
                f"timeout={timeout} stop_event_set={bool(stop_event and stop_event.is_set())}"
            )
            self.logger.debug(
                f"Threaded download callbacks: callback_set={bool(callback)} callback_remux_set={bool(callback_remux)}"
            )
            resume_state = None
            resume_mode = False
            created_at = None

            # Help type checker with initial types
            if segment_state_path:
                if os.path.exists(segment_state_path):
                    self.logger.info(f"Found segment state file: {segment_state_path}. Attempting resume.")
                else:
                    self.logger.debug(f"No segment state file found at: {segment_state_path}. Starting fresh.")

            if segment_state_path and os.path.exists(segment_state_path):
                try: # This starts resuming from previous download
                    resume_state = load_segment_state(segment_state_path)
                    resume_mode = True
                except Exception as e: # Shouldn't happen, but if it does, we just do a new download
                    self.logger.warning(f"Failed to load segment state {segment_state_path}: {e}. Starting fresh.")
                    resume_state = None
                    resume_mode = False

            if resume_mode:
                assert resume_state is not None
                segments = resume_state.get("segments") or []  # This fetches the list of segments from the resume state
                if not segments:
                    raise UnknownError("Segment state is invalid or empty.") # Shouldn't happen ;)

                segment_dir = resume_state.get("segment_dir") or segment_dir
                if not segment_dir:
                    raise UnknownError("Segment state is missing segment_dir.")

                created_at = resume_state.get("created_at")
                width = int(resume_state.get("segment_index_width") or get_segment_index_width(len(segments)))
                state_start = int(resume_state.get("start_segment", 0) or 0) # Where we start segments

                """
            Because every segment has a different binary offset, we can't just inject specific segments into specific
            parts of the file. That's why I can only start after xx successful segments.

            So, let's say 0-12 segments were successful, but 13 was not and from 14-17 everything went smooth.
            In this case, I need to start from 13 and STILL override 14-17.
                """

                if start_segment and state_start != start_segment:
                    self.logger.warning(
                        f"start_segment={start_segment} ignored; resuming from state start_segment={state_start}."
                    )

                start_segment = state_start
                m3u8_url = resume_state.get("m3u8_url") or ""
                state_quality = resume_state.get("quality", quality)
                self.logger.info(
                    f"Resume state loaded: segments={len(segments)} start_segment={start_segment} "
                    f"segment_dir={segment_dir} segment_index_width={width} created_at={created_at} "
                    f"quality={state_quality} m3u8_url={m3u8_url}"
                )

            else:
                m3u8_master = pre_resolved_m3u8_url
                self.logger.info(f"Fetching segments for quality={quality} m3u8_url_master={m3u8_master}")
                segments = await self.get_segments(quality=quality, m3u8_url_master=m3u8_master)
                total_before = len(segments)
                if start_segment > 0:
                    self.logger.debug(
                        f"Applying start_segment offset: {start_segment} (from total={total_before})"
                    )
                    segments = segments[start_segment:]
                if segment_state_path and segment_dir is None:
                    segment_dir = f"{path}.segments"
                    self.logger.debug(f"segment_dir set from state path: {segment_dir}")
                width = get_segment_index_width(len(segments)) if segment_dir else 0
                m3u8_url = m3u8_master
                state_quality = quality
                self.logger.info(
                    f"Segments ready: count={len(segments)} segment_dir={segment_dir} "
                    f"segment_index_width={width} m3u8_url={m3u8_url}"
                )

            n = len(segments) # Total amount of segments
            if n == 0:
                raise UnknownError("No segments found for this playlist.")
                # Shouldn't happen

            if segment_dir:
                os.makedirs(segment_dir, exist_ok=True) # Creates the segment directory for later resuming
                self.logger.debug(f"Segment directory ready: {segment_dir}")
            self.logger.info(f"Segment plan: total={n} segment_dir={segment_dir}")

            downloaded = [False] * n # Keeps track of total downloaded segments

            """
            We write a list with [False, False, n] where n is the value of the total amount of segments.
            This creates a lit with as many False entries as segments. Since `self.download_segment` returns a bool
            along with the data, we can use that to keep track, since we just change the bool to True for every 
            downloaded segments.
            """

            if segment_dir: # Tries to find existing segments that we already downloaded
                existing_segments = 0
                for i in range(n): # Does that for every segment
                    seg_path = segment_file_path(segment_dir, i, width) # Gets the file path
                    try:
                        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                            # if it exists, we treat it as already downloaded (makes sense)
                            downloaded[i] = True
                            existing_segments += 1
                    except Exception as exc:
                        self.logger.warning(f"Couldn't download segment: {i}, retrying later.  ->: {exc}")
                        # If something goes wrong, we treat it as not downloaded and re-fetch it later
                        downloaded[i] = False
                self.logger.info(
                    f"Existing segments detected: {existing_segments}/{n} in {segment_dir}"
                )

            progressed = sum(downloaded) # Amount of already downloaded segments
            downloaded_count = progressed
            if progressed and callback: # Does an initial callback, so that Porn Fetch can start showing the user how
                # many segments have already been downloaded
                callback(progressed, n)
            if progressed:
                self.logger.info(f"Resume progress: already_downloaded={progressed}/{n}")

            target_indices = [i for i in range(n) if not downloaded[i]] # The segments we still need to fetch
            self.logger.info(f"Target segments to download: {len(target_indices)}/{n}")

            tmp_path = f"{path}.tmp" # Creates a temporary path where we write stuff to
            cancelled = False # This is the cancellation event that stops the download
            max_seg_retries = 2 # Maximum retries to get segments
            progress_log_step = max(1, n // 20)
            next_progress_log = ((progressed // progress_log_step) + 1) * progress_log_step

            if stop_event is not None and stop_event.is_set():
                cancelled = True
                target_indices = [] # Empty list stops the download :)
                self.logger.warning("Stop event already set; cancelling before scheduling segments.")

            if target_indices:
                workers = max(1, min(max_workers, len(target_indices)))
                parts: List[bytes | None] | None = None
                next_to_write = 0
                out_fp = None
                self.logger.info(
                    f"Starting segment download pool: workers={workers} targets={len(target_indices)}"
                )

                if not segment_dir:
                    parts = [None] * n
                    out_fp = cast(Any, open(tmp_path, "wb"))
                    self.logger.debug(f"Using in-memory segment assembly. tmp_path={tmp_path}")
                else:
                    self.logger.debug(f"Writing segments to disk. segment_dir={segment_dir} tmp_path={tmp_path}")

                try:
                    # Use asyncio.gather to fetch segments concurrently instead of ThreadPoolExecutor

                    # Create a semaphore to limit concurrent requests
                    semaphore = asyncio.Semaphore(workers)

                    async def fetch_segment_with_semaphore(idx: int, url: str) -> Tuple[int, bool, bytes]:
                        async with semaphore:
                            if stop_event is not None and stop_event.is_set():
                                return idx, False, b""

                            # Handle retries inside the coroutine
                            for attempt in range(max_seg_retries + 1):
                                if stop_event is not None and stop_event.is_set():
                                    return idx, False, b""

                                try:
                                    _, segment_data, is_success = await self.download_segment(url, timeout, stop_event)
                                    if is_success and segment_data:
                                        return idx, True, segment_data
                                except Exception as exception:
                                    self.logger.error(f"Worker exception for segment {idx}: {exception}", exc_info=True)

                                if attempt < max_seg_retries:
                                    self.logger.warning(
                                        f"Segment {idx} failed; retrying {attempt + 1}/{max_seg_retries}"
                                    )
                                    # Optional short backoff delay could go here
                                else:
                                    self.logger.error(
                                        f"Segment {idx} failed after {attempt} retries."
                                    )
                            return idx, False, b""

                    tasks = [fetch_segment_with_semaphore(i, segments[i]) for i in target_indices]

                    # Use asyncio.as_completed to process results as they come in, similar to wait(FIRST_COMPLETED)
                    for coro in asyncio.as_completed(tasks):
                        if stop_event is not None and stop_event.is_set():
                            cancelled = True
                            # The remaining tasks will see the event set and exit quickly
                            continue

                        i, success, data = await coro

                        if cancelled:
                            continue

                        if success and data:
                            downloaded[i] = True # Successfully got segment, mark it as done
                            downloaded_count += 1
                            if segment_dir:
                                # Write to a temp path (good for resuming, but not I/O efficient)
                                seg_path = segment_file_path(segment_dir, i, width)
                                tmp_seg = f"{seg_path}.part"
                                # Offload segment file writing to a thread
                                def write_part(ts_path: str, t_data: bytes) -> None:
                                    with open(ts_path, "wb") as f:
                                        f.write(t_data)
                                await asyncio.to_thread(write_part, tmp_seg, data)
                                os.replace(tmp_seg, seg_path)
                            else:
                                assert parts is not None
                                parts[i] = data # Keep in memory (I/O efficient)

                            progressed += 1 # Fetched +1 segment, so we give back callback
                            if callback:
                                callback(progressed, n)
                            if progressed >= next_progress_log or progressed == n:
                                remaining = n - downloaded_count
                                self.logger.debug(
                                    f"Segment progress: processed={progressed}/{n} "
                                    f"downloaded={downloaded_count} remaining={remaining}"
                                )
                                next_progress_log += progress_log_step

                        else:
                            # Handling failure (already retried in fetch_segment_with_semaphore)
                            progressed += 1
                            if callback:
                                callback(progressed, n)
                            if progressed >= next_progress_log or progressed == n:
                                remaining = n - downloaded_count
                                self.logger.debug(
                                    f"Segment progress: processed={progressed}/{n} "
                                    f"downloaded={downloaded_count} remaining={remaining}"
                                )
                                next_progress_log += progress_log_step

                        if not segment_dir and parts is not None:
                            chunks_to_write = []
                            while next_to_write < n and parts[next_to_write] is not None:
                                if parts[next_to_write]:
                                    chunks_to_write.append(parts[next_to_write])
                                next_to_write += 1
                            if chunks_to_write:
                                # Write memory chunks to thread to prevent IO block
                                def write_chunks(fp: Any, list_of_data: List[bytes]) -> None:
                                    for c_data in list_of_data:
                                        fp.write(c_data)
                                await asyncio.to_thread(write_chunks, cast(Any, out_fp), chunks_to_write)

                finally:
                    if out_fp is not None:
                        out_fp.close()

            missing = [i for i, ok in enumerate(downloaded) if not ok] # Missing segments
            missing_urls = [segments[i] for i in missing] # Missing URLs of segments
            self.logger.info(
                "Segment download finished: downloaded=%s/%s missing=%s cancelled=%s",
            downloaded_count, n, len(missing), cancelled)
            if missing:
                sample = missing[:10]
                self.logger.error(
                    "Missing segments detected: count=%s sample=%s", len(missing), sample
                )

            report = DownloadReport(
                status= "cancelled" if cancelled else ("failed" if missing else "completed"),
                total=n,
                downloaded= n - len(missing),
                missing=missing,
                missing_urls=missing_urls,
                segment_dir=segment_dir,
                segment_state_path=segment_state_path,
                start_segment=start_segment,
                quality=quality

            )

            if cancelled: # If user cancels, we clean up stuff
                self.logger.warning(
                    f"Download cancelled. cleanup_on_stop={cleanup_on_stop} keep_segment_dir={keep_segment_dir}"
                )
                if cleanup_on_stop:
                    self._safe_remove(tmp_path)
                    if segment_dir and not keep_segment_dir:
                        self._safe_rmtree(segment_dir)

                if segment_state_path:
                    # This is the segment state that is saved as a file, this is NOT the returned report!
                    assert  isinstance(segment_state_path, str)
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = build_segment_state(
                        segments=segments,
                        missing=missing,
                        segment_dir=segment_dir,
                        segment_index_width=width if segment_dir else 0,
                        path=path,
                        quality=str(state_quality),
                        start_segment=start_segment,
                        m3u8_url=m3u8_url,
                        created_at=created_at,
                    )
                    write_segment_state(segment_state_path, state)

                if return_report:
                    missing = report.missing
                    self.logger.debug(
                        f"Returning cancelled report: downloaded={report.downloaded} missing={len(missing)}"
                    )
                    return report
                return False

            if missing:
                self.logger.error(
                    f"Download incomplete: {len(missing)} segments missing. Writing state={bool(segment_state_path)}"
                )
                self._safe_remove(tmp_path)
                if segment_state_path:
                    self.logger.info(f"Writing segment state to: {segment_state_path}")
                    state = build_segment_state(
                        segments=segments,
                        missing=missing,
                        segment_dir=segment_dir,
                        segment_index_width=width if segment_dir else 0,
                        path=path,
                        quality=str(state_quality),
                        start_segment=start_segment,
                        m3u8_url=m3u8_url,
                        created_at=created_at,
                    )
                    write_segment_state(segment_state_path, state)
                if return_report:
                    self.logger.debug(
                        f"Returning failed report: downloaded={report.downloaded} missing={len(report.missing)}"
                    )
                    return report
                return False

            if segment_dir:
                self.logger.info(
                    f"Assembling {n} segments from {segment_dir} into {tmp_path}"
                )
                def assemble_segments() -> List[int]:
                    with open(tmp_path, "wb") as out_file_path:
                        for idx in range(n):
                            segment_path = segment_file_path(segment_dir, idx, width)
                            if not os.path.exists(segment_path):
                                return [idx]
                            with open(segment_path, "rb") as seg_fp:
                                shutil.copyfileobj(seg_fp, out_file_path, length=1024 * 1024) # type: ignore[arg-type]
                    return []

                # Offload heavy IO segment assembly
                missing_assemble = await asyncio.to_thread(assemble_segments)
                if missing_assemble:
                    missing = missing_assemble

                if missing:
                    self.logger.error(
                        f"Missing segment file during assemble: index={missing[0]} segment_dir={segment_dir}"
                    )
                    self._safe_remove(tmp_path)
                    if segment_state_path:
                        self.logger.info(f"Writing segment state to: {segment_state_path}")
                        state = build_segment_state(
                            segments=segments,
                            missing=missing,
                            segment_dir=segment_dir,
                            segment_index_width=width if segment_dir else 0,
                            path=path,
                            quality=str(state_quality),
                            start_segment=start_segment,
                            m3u8_url=m3u8_url,
                            created_at=created_at,
                        )
                        write_segment_state(segment_state_path, state)
                    report.status = "failed"
                    report.missing = missing
                    report.missing_urls = [segments[i] for i in missing]
                    if return_report:
                        self.logger.debug(
                            f"Returning failed report after assemble: downloaded={report.downloaded} "
                            f"missing={len(report.missing)}"
                        )
                        return report
                    return False

            if remux:
                self.logger.info(f"Remuxing TS to MP4: input={tmp_path} output={path}")
                # Offload heavy CPU/IO bound task
                await asyncio.to_thread(self._convert_ts_to_mp4, tmp_path, path, callback_remux, ios_support)
                # This is important, because not all players can play MPEG-TS AND I want to write
                # metadata to the files, and this doesn't work without a container.
                self._safe_remove(tmp_path)
                self.logger.info(f"Remux completed: output={path}")

            else:
                self.logger.debug("Remux disabled; moving temporary file into place.")
                try:
                    os.replace(tmp_path, path) # If we don't remux, we just rename it to mp4 and treat it as done :)
                except Exception as exc: # Shouldn't happen and I also don't know what this does lol
                    self.logger.warning(f"os.replace failed: {exc}, falling back to manual copy.")
                    def manual_copy() -> None:
                        with open(path, "wb") as final_fp, open(tmp_path, "rb") as in_fp:
                            for chunk in iter(lambda: in_fp.read(1024 * 1024), b""):
                                final_fp.write(chunk)
                    await asyncio.to_thread(manual_copy)
                    self._safe_remove(tmp_path) # Remove stuff I guess

            if segment_dir and not keep_segment_dir:
                self._safe_rmtree(segment_dir) # Delete segment dir (cleanup) (optional)
            if segment_state_path: # Delete segment state (optional)
                self._safe_remove(segment_state_path)
            self.logger.info(f"Download completed successfully: path={path}")

            if return_report: # Do a report, if user asked to
                self.logger.debug(
                    f"Returning completed report: downloaded={report.downloaded} missing={len(report.missing)}"
                )
                return report
            return True
        except Exception as e:
            self.logger.exception(f"Unhandled exception in download wrapper: {e}")
            return False

    def _convert_ts_to_mp4(self, input_path: str, output_path: str,
                           callback: Callable[[int, int], None] | None = None, ios_support: bool = False) -> None:
        start_ts = time.perf_counter()
        self.logger.info("Remux start: input=%s output=%s", input_path, output_path)

        try:
            input_size = os.path.getsize(input_path)
            self.logger.debug("Remux input size: %s bytes", input_size)
        except Exception as e:
            self.logger.debug("Remux input size unavailable: %s", e)

        try:
            from av import open as av_open  # type: ignore[import-not-found]
            from av.audio.resampler import AudioResampler  # type: ignore[import-not-found]
            import av.audio.frame  # Used for runtime isinstance check
        except (ModuleNotFoundError, ImportError) as e:
            self.logger.error("PyAV import failed for remux: %s", e, exc_info=True)
            raise ModuleNotFoundError(
                f"PyAV is required for remuxing. Install with pip install av. Not supported on Termux! {e}") from e

        self.logger.debug("Opening input for remux: %s", input_path)
        input_ = av_open(input_path)
        fmt_name = (input_.format.name or "").lower()
        self.logger.info("Input format detected: %s", fmt_name or '<unknown>')

        if fmt_name == "mpegts":
            # Fix 1: Suppress the stub mismatch for av.open
            output = av_open(output_path, mode="w", format="mp4",
                             options={"movflags": "faststart"})  # type: ignore[arg-type]

            # --- VIDEO ---
            in_video = input_.streams.video[0]
            out_video = output.add_stream_from_template(template=in_video)
            self.logger.debug(
                "Video stream: codec=%s bit_rate=%s",
                getattr(in_video.codec_context, 'name', None), getattr(in_video.codec_context, 'bit_rate', None)
            )

            # --- AUDIO ---
            in_audio = next((s for s in input_.streams if s.type == "audio"), None)
            out_audio = None
            transcode_audio = False
            resampler = None

            if in_audio:
                # Fix 3: Explicitly narrow out None
                assert in_audio is not None

                # Fix 2: Cast context to AudioCodecContext so IDE knows about sample_rate and layout
                audio_ctx = cast('AudioCodecContext', in_audio.codec_context)

                copy_ok = {"aac"} if ios_support else {"aac", "alac", "mp3"}
                codec_name = (audio_ctx.name or "").lower()
                sample_rate = audio_ctx.sample_rate or 0
                layout_name = audio_ctx.layout.name if getattr(audio_ctx, "layout", None) else "unknown"

                self.logger.debug(
                    "Audio stream: codec=%s sample_rate=%s layout=%s", codec_name, sample_rate, layout_name
                )

                if codec_name in copy_ok:
                    out_audio = output.add_stream_from_template(template=in_audio)
                    self.logger.info("Audio codec MP4-compatible; remuxing without transcoding.")
                else:
                    transcode_audio = True
                    sample_rate = audio_ctx.sample_rate or 48000
                    layout = audio_ctx.layout.name if getattr(audio_ctx, "layout", None) else "stereo"

                    out_audio = output.add_stream("aac", rate=sample_rate)
                    self.logger.info("Transcoding audio to AAC: sample_rate=%s layout=%s"), sample_rate, layout

                    try:
                        out_audio.layout = layout
                    except Exception as exc:
                        self.logger.warning("Exception in getting audio layout (doesn't matter): %s", exc)
                        pass

                    resampler = AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            else:
                self.logger.info("No audio stream detected; remuxing video only.")

            # --- DEMUX ---
            demux_streams = [in_video] + ([in_audio] if in_audio else [])
            packets = input_.demux(demux_streams)

            try:
                total = os.path.getsize(input_path)
            except Exception as exc:
                self.logger.warning("Exception while getting path size for demuxing progress??? %s", exc)
                total = 100

            self.logger.info("Demuxing packets: total_bytes=%s", total)
            progress_step = max(1, total // 10) if total else 0
            next_progress_log = progress_step if progress_step else 0
            current_progress = 0

            for idx, packet in enumerate(packets):
                pkt_size = getattr(packet, "size", 0) or 0
                current_progress += pkt_size

                if packet.dts is None:
                    if callback:
                        callback(current_progress, total)
                    continue

                if packet.stream == in_video:
                    packet.stream = out_video
                    output.mux(packet)

                elif in_audio and packet.stream == in_audio:
                    if not transcode_audio:
                        packet.stream = out_audio
                        output.mux(packet)
                    else:
                        assert out_audio is not None
                        for frame in packet.decode():
                            # Fix 4: Ensure the frame is recognized as an AudioFrame
                            if not isinstance(frame, av.audio.frame.AudioFrame):
                                continue

                            frames = resampler.resample(frame) if resampler else [frame]
                            for f in frames:
                                for enc_pkt in out_audio.encode(f):
                                    output.mux(enc_pkt)

                if callback:
                    callback(current_progress, total)
                if progress_step and current_progress >= next_progress_log:
                    self.logger.debug("Remux progress: bytes=%s/%s", current_progress, total)
                    next_progress_log += progress_step

            if transcode_audio and out_audio:
                self.logger.debug("Flushing AAC encoder.")
                for enc_pkt in out_audio.encode(None):
                    output.mux(enc_pkt)

            input_.close()
            output.close()
            elapsed = time.perf_counter() - start_ts

            try:
                out_size = os.path.getsize(output_path)
                self.logger.info("Remux complete: output=%s size=%s bytes elapsed=%s.2f", output_path, out_size,
                                 elapsed)
            except Exception as e:
                self.logger.info("Remux complete: output=%s elapsed=%s.2fs (size unavailable: %s)", output_path,
                                 elapsed, e)

        else:
            self.logger.info("Stream seems to be already in MP4! Skipping remux...")
            os.rename(input_path, output_path)
            elapsed = time.perf_counter() - start_ts
            self.logger.info("Remux skipped; file moved. elapsed=%s.2f", elapsed)

    async def legacy_download(self, url: str, configuration: DownloadConfigRAW) -> bool:
        """
        Download a file using streaming with stall tolerance and resume.
        Supports fast concurrent range downloading if the server supports it and allow_multipart is True.
        Assumes self.session is an AsyncSession.
        """
        path = configuration.path
        max_retries = configuration.max_retries
        read_timeout = configuration.read_timeout
        stop_event = configuration.stop_event
        allow_multipart = configuration.allow_multipart
        callback = configuration.callback
        chunk_size = configuration.chunk_size
        max_workers = configuration.max_workers

        self.logger.info(
"""Legacy download start: url=%s path=%s
max_retries=%s read_timeout=%s
stop_event_set=%s
allow_multipart=%s""", url, path, max_retries, read_timeout, bool(stop_event and stop_event.is_set()),
        allow_multipart)

        if stop_event is not None and stop_event.is_set():
            self.logger.warning("Stop event already set; cancelling legacy download.")
            raise DownloadCancelled("Download cancelled.")

        # Ensure session is initialized
        if self.session is None:
            self.initialize_session()
        session = self.session
        assert session is not None

        progress_bar = None
        if callback is None:
            progress_bar = Callback()
            self.logger.debug("legacy_download: no callback provided, using default progress bar")

        timeout = read_timeout

        # 1. Check if the server supports Range requests and get file size (if multipart is allowed)
        file_size = 0
        accept_ranges = ""

        if allow_multipart:
            # We MUST request uncompressed content for range downloads, otherwise:
            # 1) Content-Length from HEAD reflects the compressed size, not the real file size.
            # 2) Mid-file Range requests on compressed streams cause libcurl error 61
            #    ("incorrect header check") because partial gzip lacks a valid header.
            no_compress = {"Accept-Encoding": "identity"}
            try:
                head_resp = await session.head(url, timeout=timeout, allow_redirects=True, headers=no_compress)
                if head_resp.status_code == 405:  # Method Not Allowed, fallback to streaming GET
                    head_resp_stream = await session.request("GET", url, timeout=timeout, allow_redirects=True,
                                                             stream=True, headers=no_compress)
                    file_size = int(head_resp_stream.headers.get("Content-Length", 0))
                    accept_ranges = head_resp_stream.headers.get("Accept-Ranges", "")
                else:
                    file_size = int(head_resp.headers.get("Content-Length", 0))
                    accept_ranges = head_resp.headers.get("Accept-Ranges", "")
            except Exception as e:
                self.logger.warning("Failed to fetch HEAD info for concurrent check: %s.", e)

        # 2. Execute Fast Multipart Download if supported and allowed
        if allow_multipart and file_size > 0 and accept_ranges == "bytes":
            self.logger.info("Server supports Range requests. Starting fast multipart download"
                             "or %s bytes.", file_size)

            # Pre-allocate file
            def allocate_file() -> None:
                if not os.path.exists(path):
                    with open(path, "wb") as file_alloc:
                        file_alloc.truncate(file_size)
                elif os.path.getsize(path) != file_size:
                    # File exists but size mismatch, truncate to correct size
                    with open(path, "r+b") as file_alloc_size:
                        file_alloc_size.truncate(file_size)
            await asyncio.to_thread(allocate_file)

            # We will use an array to track progress of chunks
            # A chunk map: {chunk_index: bytes_downloaded}
            chunk_progress = {}
            total_downloaded = [0]  # List to allow modification in inner func
            # Determine chunk sizes based on file size, but keep reasonable bounds
            # For massive files, don't create 10,000 workers.
            target_chunk_size = max(chunk_size, min(10 * 1024 * 1024, file_size // 10)) # Between 1MB and 10MB

            semaphore = asyncio.Semaphore(max_workers)

            async def download_chunk(start_chunk: int, end_chunk: int, chunk_idx_now: int) -> bool:
                nonlocal total_downloaded
                headers_chunk = {"Range": f"bytes={start_chunk}-{end_chunk}", "Accept-Encoding": "identity"}
                chunk_progress[chunk_idx_now] = 0

                for attempt_chunk in range(max_retries + 1):
                    if stop_event is not None and stop_event.is_set():
                        return False

                    try:
                        async with semaphore:
                            resp = await cast(Any, session).request(
                                "GET", url, headers=headers_chunk, timeout=timeout, allow_redirects=True, stream=True
                            )
                            resp.raise_for_status()

                            # Open file once for this chunk download attempt
                            file = await asyncio.to_thread(lambda: open(path, "rb+"))
                            try:
                                await asyncio.to_thread(file.seek, start_chunk + chunk_progress[chunk_idx_now])
                                async for data in resp.aiter_content():
                                    if stop_event is not None and stop_event.is_set():
                                        return False

                                    await asyncio.to_thread(cast(Any, file).write, data)

                                    data_len = len(data)
                                    chunk_progress[chunk_idx_now] += data_len
                                    total_downloaded[0] += data_len

                                    if callback:
                                        callback(total_downloaded[0], file_size)
                                    elif progress_bar:
                                        progress_bar.text_progress_bar(downloaded=total_downloaded[0], total=file_size)
                            finally:
                                await asyncio.to_thread(file.close)

                            return True # Chunk success

                    except Exception as exc:
                        if attempt_chunk < max_retries:
                            self.logger.warning("Chunk %s failed (attempt %s/%s): %s",
                                                chunk_idx_now, attempt_chunk + 1, max_retries, exc)
                            # Reset progress for this chunk before retry
                            total_downloaded[0] -= chunk_progress[chunk_idx_now]
                            chunk_progress[chunk_idx_now] = 0
                            await asyncio.sleep(1 * attempt_chunk)
                        else:
                            self.logger.error("Chunk %s permanently failed: %s", chunk_idx_now, exc, exc_info=True)
                            return False
                return False

            tasks = []
            chunk_idx = 0
            for start in range(0, file_size, target_chunk_size):
                end = min(start + target_chunk_size - 1, file_size - 1)
                tasks.append(download_chunk(start, end, chunk_idx))
                chunk_idx += 1

            results = await asyncio.gather(*tasks)

            if progress_bar:
                # We set it to None instead of del to avoid analyzer confusion about potential unassigned reference
                progress_bar = None

            if stop_event is not None and stop_event.is_set():
                raise DownloadCancelled("Download cancelled.")

            if not all(results):
                raise NetworkRequestError("One or more chunks failed to download completely.")

            self.logger.info("Fast multipart download complete: path=%s", path)
            return True

        # 3. Fallback to standard linear streaming download
        if not allow_multipart:
            self.logger.info("allow_multipart=False. Forcing linear streaming download.")
        else:
            self.logger.info("Server does not support Range requests or size is 0. Falling back to linear streaming.")

        downloaded_so_far = 0
        attempt = 0
        etag = None

        while True:
            if stop_event is not None and stop_event.is_set():
                self.logger.warning("Stop event set; cancelling legacy download.")
                raise DownloadCancelled("Download cancelled.")
            headers = {}
            if downloaded_so_far:
                headers["Range"] = f"bytes={downloaded_so_far}-"

            try:
                response = await cast(Any, session).request(
                    "GET", url, headers=headers, allow_redirects=True, timeout=timeout, stream=True
                )
                if downloaded_so_far and response.status_code == 200:
                    self.logger.warning("Server ignored Range request; restarting download from scratch.")
                    downloaded_so_far = 0
                response.raise_for_status()

                etag_cur = response.headers.get("ETag")
                if etag is None:
                    etag = etag_cur
                elif etag_cur and etag_cur != etag:
                    raise RuntimeError("Remote content changed during download")

                total = None
                cr = response.headers.get("Content-Range")
                if cr and "/" in cr:
                    try: total = int(cr.rsplit("/", 1)[1])
                    except ValueError: pass
                if total is None:
                    try: total = int(response.headers.get("Content-Length", "0")) or None
                    except ValueError: pass

                # Fix fallback if total size is still missing
                if total is None:
                    total = 0

                mode = "ab" if downloaded_so_far else "wb"
                f = await asyncio.to_thread(cast(Any, open), path, mode)
                try:
                    await asyncio.to_thread(f.seek, 0, 2)  # Move to EOF
                    async for chunk in response.aiter_content():
                        if stop_event is not None and stop_event.is_set():
                            raise DownloadCancelled("Download cancelled.")
                        if not chunk:
                            continue
                        await asyncio.to_thread(f.write, chunk)
                        downloaded_so_far += len(chunk)

                        if callback:
                            callback(downloaded_so_far, total)
                        elif progress_bar:
                            progress_bar.text_progress_bar(downloaded=downloaded_so_far, total=total)
                finally:
                    await asyncio.to_thread(f.close)

                if progress_bar:
                    progress_bar = None
                self.logger.info("Legacy download complete: bytes=%s path=%s", downloaded_so_far, path)
                return True
            except RequestsError as e:
                err_str = str(e).lower()
                if "timeout" in err_str or "read" in err_str:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    backoff = min(2 ** attempt, 30)
                    self.logger.warning("Read timeout; retrying %s/%s in %s", attempt, max_retries,  backoff)
                    if stop_event is not None and stop_event.wait(backoff):
                        raise DownloadCancelled("Download cancelled.") from e
                    else:
                        await asyncio.sleep(backoff)
                    continue
                else:
                    raise NetworkRequestError(f"Stream for: {url} was closed or failed: {e}") from e
            except DownloadCancelled:
                raise
            except Exception as exc:
                error = traceback.format_exc()
                raise NetworkRequestError(f"Unknown error for: {url} -->: {error}") from exc

        return False
