"""Small URL operations shared by provider pagination and profile adapters."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def update_url_query(url: str, **changes: str | int | None) -> str:
    """Replace named query keys without losing filters or treating anchors as HTTP."""
    parsed = urlsplit(url)
    pairs = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key not in changes]
    pairs.extend((key, str(value)) for key, value in changes.items() if value is not None)
    return urlunsplit(parsed._replace(query=urlencode(pairs), fragment=""))


def replace_url_path(url: str, path: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit(parsed._replace(path=path, fragment=""))


def append_url_path(url: str, *parts: str | int) -> str:
    path = urlsplit(url).path.rstrip("/")
    return replace_url_path(url, path + "/" + "/".join(str(part).strip("/") for part in parts))


def profile_base_url(url: str, *, depth: int = 2) -> str:
    """Remove a provider's tab/page suffix, retaining its host, slug and filters."""
    parsed = urlsplit(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < depth:
        raise ValueError("An individual profile URL with a name or ID is required")
    return update_url_query(replace_url_path(url, "/" + "/".join(parts[:depth])), page=None)


def same_site_url(original: str, resolved: str) -> bool:
    """Accept response metadata on the same site, including localized subdomains."""
    before, after = urlsplit(original), urlsplit(resolved)
    if after.scheme not in {"http", "https"} or not before.hostname or not after.hostname:
        return False
    # The supported providers use these two-label domains, except RedTube Brazil.
    depth = 3 if before.hostname.endswith(".com.br") else 2
    return before.hostname.split(".")[-depth:] == after.hostname.split(".")[-depth:]


class FetchedText(str):
    """A string that retains response identity through text caches and parsers."""

    url: str

    def __new__(cls, content: str, *, url: str):
        value = super().__new__(cls, content)
        value.url = url
        return value
