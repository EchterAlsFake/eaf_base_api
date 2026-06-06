import os
import json
from .type_hints import DownloadState
from datetime import timezone, datetime
from curl_cffi.requests import Response
from typing import Dict, Any, cast, List
from email.utils import parsedate_to_datetime


def get_segment_index_width(total: int) -> int:
    return max(6, len(str(max(0, total - 1))))


def segment_file_path(segment_dir, index: int, width: int) -> str:
    return os.path.join(segment_dir, f"seg_{index:0{width}d}.ts")


def write_segment_state(state_path: str, state: DownloadState) -> None:
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(state, fp, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def load_segment_state(state_path: str) -> Dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as fp:
        return cast(Dict[str, Any], json.load(fp))


def build_segment_state(
    *,
    segments: List[str],
    missing: List[int],
    segment_dir: str | None,
    segment_index_width: int,
    path: str,
    quality: str,
    start_segment: int,
    m3u8_url: str | None,
    created_at: str | None = None
) -> DownloadState:
    now = datetime.now(timezone.utc).isoformat()
    state = DownloadState(
        version=1,
        created_at=created_at or now,
        updated_at=None,
        m3u8_url=m3u8_url,
        quality=quality,
        output_path=path,
        segment_dir=segment_dir,
        segment_index_width=segment_index_width,
        start_segment=start_segment,
        total=len(segments),
        missing=missing,
        segments=segments
    )
    return state


def truncate(name: str, max_bytes: int = 245) -> str:  # only 245, because we need to append .mp4
    """
    Some websites have titles that are so long (looking at you missav.ws) that you can't name a file like
    that, and thus we need to make sure the file name doesn't exceed the OS limits lol
    """
    encoded = name.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        # Ensure not to cut in middle of a UTF-8 sequence
        while encoded[-1] & 0b11000000 == 0b10000000:
            encoded = encoded[:-1]
        return cast(bytes, cast(Any, encoded)).decode("utf-8", errors="ignore")
    return name


def str_to_bool(value: str) -> bool:
    # Some function that I have for some reason I don't know if this has ever been used lmao
    """
    This function is needed for the ArgumentParser for the CLI version of my APIs. It basically maps the
    booleans for the --no-title option to valid Python boolean values.
    """
    val = value.lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def format_headers_for_log(headers: Any) -> Dict[str, str]:
    """Redact sensitive headers but keep enough signal for debugging."""
    sensitive = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-csrf-token",
        "x-xsrf-token",
    }
    out: Dict[str, str] = {}
    for key, value in headers.items():
        l_key = key.lower()
        if l_key in sensitive:
            if l_key == "cookie":
                parts = [p.split("=", 1)[0].strip() for p in str(value).split(";") if p.strip()]
                value = f"<redacted:{','.join(parts)}>" if parts else "<redacted>"
            else:
                value = "<redacted>"
        if key in out:
            out[key] = f"{out[key]}, {value}"
        else:
            out[key] = str(value)
    return out


def response_body_preview(logger, response: Response, max_bytes: int = 512) -> str:
    try:
        raw = response.content[:max_bytes]
    except Exception as e:
        return f"<failed to read body: {e}>"
    if not raw:
        return "<empty>"
    enc = getattr(response, "encoding", None) or "utf-8"
    try:
        text = cast(bytes, cast(Any, raw)).decode(enc, errors="replace")
    except Exception as exc:
        logger.error(f"There was an error while decoding text from the response body preview: {exc}")
        text = cast(bytes, cast(Any, raw)).decode("utf-8", errors="replace")
    return text.replace("\r", "\\r").replace("\n", "\\n")


def parse_retry_after(logger, response: Response) -> float | None:
    """Parse Retry-After (seconds or http-date) into seconds; None if not present/invalid."""
    v = response.headers.get("Retry-After")
    if not v:
        return None
    try:
        # numeric seconds
        return float(v)
    except ValueError:
        try:
            dt = parsedate_to_datetime(v)
            # Convert to seconds from now
            delta = (dt - dt.now(dt.tzinfo)).total_seconds()
            # clamp: negative -> 0
            return max(0.0, delta)
        except Exception as exc:
            logger.warning(f"Couldn't parse retry after in 429 error: {exc}")
            return None


def log_precondition_failed(logger, response: Response, attempt: int) -> None:
    req = response.request
    try:
        req_headers = format_headers_for_log(req.headers) if req is not None else {}
    except Exception as e:
        req_headers = {"<error>": f"failed to format request headers: {e}"}

    try:
        resp_headers = format_headers_for_log(response.headers)
    except Exception as e:
        resp_headers = {"<error>": f"failed to format response headers: {e}"}

    try:
        cond_headers = [
            k for k in req.headers.keys() if k.lower().startswith("if-")
        ] if req is not None else []
    except Exception as exc:
        logger.warning(f"Could not get the conditional headers: {exc}")
        cond_headers = []

    cond_note = f" conditional_headers={cond_headers}" if cond_headers else ""
    body_preview = response_body_preview(logger=logger, response=response)

    logger.warning(
        "HTTP 412 precondition failed (attempt %d) for %s %s.%s request_headers=%s response_headers=%s body_preview=%s",
        attempt + 1,
        getattr(req, "method", "UNKNOWN") if req is not None else "UNKNOWN",
        response.url,
        cond_note,
        req_headers,
        resp_headers,
        body_preview,
    )
