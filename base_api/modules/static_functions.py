import os
import re
import math
import json
from .type_hints import DownloadState
from datetime import timezone, datetime
from curl_cffi.requests import Response
from typing import Dict, Any, cast, List, Callable, Tuple, Union
from email.utils import parsedate_to_datetime


HEIGHT_FROM_URI = re.compile(r'(?<!\d)(\d{3,4})[pP](?!\d)')  # e.g., 1080p, 720P


def eval_flags(flags: list[int]) -> int:
    """
    Evaluate flags.

    Args:
        flags (list[int]): List of flags arguments.

    Returns:
        int: The flag(s) value.
    """

    if len(flags):
        return flags[0]

    return 0


def subc(*args: Any) -> Callable[..., Any]:
    """
    Compile a substraction regex and apply its replacement to each call.

    Returns:
        Callable: Wrapped regex callable.
    """

    *flags, pattern, repl = args
    flags_val = eval_flags(flags)

    regex = re.compile(pattern, flags_val)

    def wrapper(*args_: Any) -> Any:
        return regex.sub(repl, *args_)

    return wrapper

parse_challenge = subc(re.DOTALL, r'(?:var )|(?:/\*.*?\*/)|\s|\n|\t|(?:n;)', '') # Parse challenge syntax
other_challenge = subc(re.DOTALL, r'(if.*?&1\)|else)', r'\1:'                  ) # Convert challenge syntax



def least_factors(n: int) -> int:
    """
    Returns the least factor of a number.
    """
    if n <= 0:
        return 0
    if n % 2 == 0:
        return 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return i
    return n


def normalize_quality_value(quality: Union[str, int]) -> Union[str, int]:
    """
    quality: represents the quality value that should be normalized
    """
    if isinstance(quality, int):
        return quality # If the quality value is already an int, just return it directly

    quality = str(quality).lower().strip() # Convert to string, lower and remove white spaces

    if quality in {"best", "half", "worst"}:
        return quality # best, half and worst are also accepted values and will be further resolved in other functions

    m = re.search(r'(\d{3,4})', quality) # Search for int values that fit 144p-2160p values. return as int
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid quality: {quality}")


def choose_quality_from_list(available: List[str | int], target: Union[str, int]) -> int:
    # available like ["240", "360", "480", "720", "1080"] (Can also be unsorted)
    available_ints = sorted({int(x) for x in available}) # -> [144, 240, etc...]
    if isinstance(target, str):
        if target == "best":
            return available_ints[-1] # Return the last index, this represents the best quality [144,360,720] -1 = 720
        if target == "worst":
            return available_ints[0] # Return the first index, this represents the worst quality [144,360,720] 0 = 144
        if target == "half":
            return available_ints[len(available_ints) // 2] # Divides all options by 2 to find the middle, rounds up
        raise ValueError("Invalid label.")

    # numeric: highest ≤ target, else closest
    le = [h for h in available_ints if h <= target]
    # This works by iterating over the available qualities until the target is reached. It creates a new list with these
    # qualities and returns the last index, as the last index in this case must be maximum best quality before we go over
    # the specified one.

    if le:
        return le[-1] # Returns the stuff
    # fallback closest (ties -> higher)
    # This happens if the user for example specified 144 as the quality, but only 240+ is available.
    return available_ints[0]



def height_from_variant(variant: Any) -> int | None:
    """Extract height from a variant:
    1) stream_info.resolution (w, h)
    2) URI pattern like .../720p/...
    """
    if getattr(variant, "stream_info", None) and variant.stream_info.resolution:
        _, h = variant.stream_info.resolution  # (width, height)
        return int(h) # -> returns the height of a variant of a m3u8 master playlist

    # Fallback to search with a regex pattern
    if variant.uri:
        m = HEIGHT_FROM_URI.search(variant.uri)
        if m:
            return int(m.group(1))

    # If nothing is found, though this shouldn't happen
    return None

def is_video_playlist(variant: Any) -> bool:
    """Filter out I-frames/audio-only playlists."""
    # m3u8 lib sometimes sets is_iframe if EXT-X-I-FRAME-STREAM-INF is present.
    if getattr(variant, "is_iframe", False):
        return False

    # If codecs known and contain only audio (mp4-a, ac-3, ec-3, etc.)
    codecs = getattr(variant.stream_info, "codecs", None) if getattr(variant, "stream_info", None) else False
    if codecs:
        # very light heuristic: if no video codec substring, probably audio-only.
        # video: avc1, hvc1, hev1, vp9, av01, dvh
        assert isinstance(codecs, str)
        if not any(v in codecs.lower() for v in ("avc1", "hvc1", "hev1", "av01", "vp9", "dvh")):
            return False

    return True

def collect_variants(master: Any) -> List[Dict[str, Any]]:
    """Normalize playlist variants to a comparable list."""
    items: List[Dict[str, Any]] = []
    for v in master.playlists:
        if not is_video_playlist(v):
            continue

        h = height_from_variant(v)
        bw = getattr(v.stream_info, "bandwidth", 0) if getattr(v, "stream_info", None) else 0
        fr = getattr(v.stream_info, "frame_rate", 0.0) if getattr(v, "stream_info", None) else 0.0
        items.append({
            "uri": v.uri,
            "height": h,                 # may be None
            "bandwidth": int(bw or 0),
            "frame_rate": float(fr or 0.0),
            "resolution": getattr(v.stream_info, "resolution", None) if getattr(v, "stream_info", None) else None,
            "raw": v
        })
    return items

def pick_by_label(variants: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """best / worst / half based on a combined rank by (height, bandwidth)."""
    # rank by height first, then bandwidth as tiebreaker
    def key_fn(v: Dict[str, Any]) -> Tuple[int, int]:
        return v["height"] or 0, v["bandwidth"]
    ordered = sorted(variants, key=key_fn)

    if not ordered:
        raise ValueError("No video variants available in master playlist.")

    elif label == "worst":
        return ordered[0]
    elif label == "half":
        return ordered[len(ordered)//2]
    elif label == "best":
        return ordered[-1]
    else:
        raise ValueError("Invalid quality label.")


def pick_by_height(variants: List[Dict[str, Any]], target: int) -> Dict[str, Any]:
    """Choose the highest height ≤ target; else closest by absolute diff (ties -> higher)."""
    with_height = [v for v in variants if v["height"] is not None]
    if with_height:
        # Prefer height <= target
        below_eq = [v for v in with_height if v["height"] <= target]
        if below_eq:
            # Among same height, prefer higher bandwidth then higher fps
            best = sorted(below_eq, key=lambda v: (v["height"], v["bandwidth"], v["frame_rate"]))[-1]
            return best

        # Fallback: closest by absolute diff; ties -> higher height
        def diff_key(v: Dict[str, Any]) -> Tuple[int, int, int, float]:
            return abs((v["height"] or 0) - target), -(v["height"] or 0), v["bandwidth"], v["frame_rate"]
        return sorted(with_height, key=diff_key)[0]

    # If we have no heights at all, fall back to bandwidth ranking
    return sorted(variants, key=lambda v: v["bandwidth"])[-1]


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


def strip_title(title: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to be safe across Windows, macOS, Linux, and Android.
    Replaces or strips illegal characters and trims to a safe length.
    """

    # Reserved characters on Windows + `/` for Unix/macOS
    illegal_chars = r'[<>:"/\\|?*\x00-\x1F]'
    sanitized = re.sub(illegal_chars, "_", title)

    # Strip invisible zero-width characters
    sanitized = re.sub(r'[\u200B-\u200D\uFEFF]', '', sanitized)

    # Strip trailing periods or spaces (Windows)
    sanitized = sanitized.rstrip(" .")

    # Prevent reserved Windows filenames
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    name_only = sanitized.split('.')[0].upper()
    if name_only in reserved_names:
        sanitized = f"_{sanitized}"

    # Trim to max length
    return sanitized[:max_length]