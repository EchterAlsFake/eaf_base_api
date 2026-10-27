"""Small shared request and download helpers for the provider APIs."""

import copy
import json
import os
from functools import wraps

from base_api.modules import errors
from base_api.modules.logger import class_name, get_logger, log_context
from base_api.modules.static_functions import strip_title
from base_api.modules.urls import same_site_url


async def fetch_content(core, url: str, *, logger, owner=None, get_json=False,
                        error_types=errors, not_found_error=None, access_denied_error=None):
    """Fetch text/JSON and translate common transport errors, keeping the cause.

    ``error_types`` allows existing provider exception imports to remain valid.
    Provider-specific fallback requests stay in the provider itself.
    """
    logger = get_logger(logger.name, owner if owner is not None else core)
    with log_context(owner, url):
        try:
            content = await core.fetch_text(url)
            resolved = getattr(content, "url", None)
            if (isinstance(resolved, str) and owner is not None
                    and getattr(owner, "url", None) == url and same_site_url(url, resolved)):
                owner.resolved_url = resolved
            return json.loads(content, strict=False) if get_json else content
        except errors.DownloadCancelled:
            raise
        except Exception as error:
            logger.exception("Request failed for %s: %s", url, error)
            error_type = None
            if isinstance(error, errors.HTTPStatusError):
                if error.status_code == 404:
                    error_type = not_found_error or error_types.NotFound
                elif error.status_code == 410:
                    # Keep ResourceGone detectable without losing HTTP status information.
                    raise
                else:
                    error_type = error_types.NetworkError
            elif isinstance(error, errors.NetworkRequestError):
                error_type = error_types.NetworkError
            elif isinstance(error, (errors.InvalidProxy, errors.ProxySSLError)):
                error_type = error_types.ProxyError
            elif isinstance(error, errors.BotProtectionDetected):
                error_type = error_types.BotDetection
            elif isinstance(error, errors.UnknownError):
                error_type = error_types.UnknownNetworkError
            elif isinstance(error, errors.AccessDeniedError):
                error_type = access_denied_error
            if error_type is None:
                raise
            name = class_name(owner if owner is not None else core)
            raise error_type(f"Request failed in {name} for {url}: {error}",
                             url=url, class_name=name, api=logger.name.split('.')[0]) from error


def download_errors(error_type=errors.DownloadFailed):
    """Give every media download the same logging, context, and cancellation rules."""
    def decorate(function):
        @wraps(function)
        async def wrapped(media, *args, **kwargs):
            name = class_name(media)
            api = function.__module__.split('.')[0]
            logger = get_logger(function.__module__, media)
            with log_context(media, media.url):
                try:
                    result = await function(media, *args, **kwargs)
                    if result is False:
                        raise error_type("Downloader returned False")
                    return result
                except errors.DownloadCancelled:
                    raise
                except errors.VideoUnavailable as error:
                    error.url, error.class_name, error.api = media.url, name, api
                    logger.exception("Video unavailable: %s", error)
                    raise
                except Exception as error:
                    logger.exception("Download failed: %s", error)
                    raise error_type(f"Download failed in {name} ({api}) for {media.url}: {error}",
                                     url=media.url, class_name=name, api=api) from error
        return wrapped
    return decorate


def prepare_download_config(configuration, title: str | None):
    """Copy a caller's configuration and apply the common output filename rule."""
    # Keep callbacks and stop_event shared with the caller; only the config's
    # own attributes (path, stream URL, default callback) are changed.
    result = copy.copy(configuration)
    if not result.no_title:
        clean_title = strip_title(title) if title else "video"
        result.path = os.path.join(result.path, f"{clean_title}.mp4")
    return result


async def download_hls(media, configuration):
    """Load HLS metadata and dispatch using an independent download configuration."""
    await media.load_fields("title", "m3u8_base_url")
    if not media.m3u8_base_url:
        raise errors.DownloadFailed("No HLS stream URL available")
    config = prepare_download_config(configuration, media.title or "video")
    config.m3u8_base_url = media.m3u8_base_url
    return await media.core.download(configuration=config)
