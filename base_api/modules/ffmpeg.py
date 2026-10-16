"""FFmpeg binary discovery, MPEG-TS to MP4 remuxing, and MP4 metadata tagging."""
from __future__ import annotations

import inspect
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from base_api.modules.config import config
from base_api.modules.errors import MetadataWriteError

logger = logging.getLogger("base_api.ffmpeg")


def _find_original_executable() -> Path | None:
    """Return the true executable path, ignoring onefile or temp extractions."""
    try:
        import __compiled__  # type: ignore[import-not-found]
        orig = getattr(__compiled__, "original_argv0", None)
        if orig and Path(orig).exists():
            return Path(orig).resolve()
    except (ImportError, OSError, TypeError):
        pass

    if appimage := os.environ.get("APPIMAGE"):
        if Path(appimage).exists():
            return Path(appimage).resolve()

    if sys.argv and sys.argv[0] and Path(sys.argv[0]).exists():
        return Path(sys.argv[0]).resolve()

    return None


def _is_executable_file(candidate: Path) -> bool:
    return candidate.is_file() and (sys.platform == "win32" or os.access(candidate, os.X_OK))


def find_ffmpeg(
    ffmpeg_bin: Path | str | None = None,
    *,
    original_executable: Path | None = None,
    extra_roots: tuple[Path, ...] = (),
) -> Path | None:
    """Discover an explicit, bundled, or system FFmpeg executable."""
    binary_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    # 1. Explicit argument, runtime config, or environment variable override
    for explicit in (
        ffmpeg_bin,
        getattr(config, "ffmpeg_path", None),
        os.environ.get("PORN_FETCH_FFMPEG_BINARY"),
        os.environ.get("EAF_FFMPEG_BINARY"),
    ):
        if explicit:
            candidate = Path(explicit).expanduser()
            if _is_executable_file(candidate):
                return candidate.resolve()

    candidates: list[Path] = []

    # 2. Paths relative to current running executable
    orig = original_executable if original_executable is not None else _find_original_executable()
    if orig is not None:
        exec_dir = orig.parent
        candidates.append(exec_dir / "third_party" / binary_name)
        candidates.append(exec_dir / binary_name)
        # macOS App bundle: Contents/MacOS/
        candidates.append(exec_dir / "Contents" / "MacOS" / "third_party" / binary_name)
        candidates.append(exec_dir / "Contents" / "MacOS" / binary_name)
        if exec_dir.name == "MacOS" and exec_dir.parent.name == "Contents":
            candidates.append(exec_dir.parent.parent.parent / "third_party" / binary_name)

    # 3. Project / caller roots and working directory
    roots: list[Path] = list(extra_roots)
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[2])
    if sys.executable:
        roots.append(Path(sys.executable).resolve().parent)

    for root in roots:
        candidates.append(root / "third_party" / binary_name)
        candidates.append(root / binary_name)

    for cand in candidates:
        if _is_executable_file(cand):
            return cand.resolve()

    # 4. System PATH fallback
    if in_path := shutil.which(binary_name) or shutil.which("ffmpeg"):
        p = Path(in_path)
        if p.is_file():
            return p.resolve()

    return None


def has_ffmpeg(
    ffmpeg_bin: Path | str | None = None,
    *,
    original_executable: Path | None = None,
    extra_roots: tuple[Path, ...] = (),
) -> bool:
    """Return True if an FFmpeg binary is available on the system or bundled."""
    return (
        find_ffmpeg(
            ffmpeg_bin,
            original_executable=original_executable,
            extra_roots=extra_roots,
        )
        is not None
    )


def _safe_getattr(obj: Any, name: str) -> Any:
    """Safely read an attribute without triggering unawaited coroutine warnings or lazy-load errors."""
    try:
        val = getattr(obj, name, None)
    except Exception:
        return None
    if inspect.iscoroutine(val):
        val.close()
        return None
    if callable(val):
        return None
    return val


def _format_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        name = value.get("name") or value.get("username") or value.get("title")
        return _format_metadata_value(name)
    name_attr = _safe_getattr(value, "name") or _safe_getattr(value, "username")
    if name_attr is not None and not isinstance(value, (str, bytes)):
        return _format_metadata_value(name_attr)
    text = str(value).strip()
    return text if text else None


def _extract_metadata_fields(data: Any) -> dict[str, str]:
    """Extract normalized FFmpeg metadata key-value pairs from an object or mapping."""
    if data is None:
        return {}

    if isinstance(data, Mapping):
        raw_fields = {
            "title": data.get("title"),
            "artist": data.get("artist") if "artist" in data else data.get("author"),
            "genre": data.get("genre", "XXX"),
            "date": data.get("date") if "date" in data else data.get("publish_date"),
        }
        for key, val in data.items():
            if key not in {"title", "artist", "author", "genre", "date", "publish_date", "thumbnail_data", "thumbnail"}:
                raw_fields[str(key)] = val
    else:
        artist = (
            _safe_getattr(data, "artist")
            or _safe_getattr(data, "author")
            or _safe_getattr(data, "author_information")
        )
        raw_fields = {
            "title": _safe_getattr(data, "title"),
            "artist": artist,
            "genre": _safe_getattr(data, "genre") or "XXX",
            "date": _safe_getattr(data, "date") or _safe_getattr(data, "publish_date"),
        }

    resolved: dict[str, str] = {}
    for key, val in raw_fields.items():
        formatted = _format_metadata_value(val)
        if formatted is not None:
            resolved[key] = formatted
    return resolved


def _extract_thumbnail_bytes(data: Any, explicit_thumbnail: bytes | Path | str | None = None) -> bytes | None:
    candidate = explicit_thumbnail
    if candidate is None and data is not None:
        if isinstance(data, Mapping):
            candidate = data.get("thumbnail_data") or data.get("thumbnail")
        else:
            candidate = _safe_getattr(data, "thumbnail_data")

    if isinstance(candidate, (str, Path)):
        p = Path(candidate)
        if p.is_file():
            return p.read_bytes()
        return None

    if isinstance(candidate, bytes) and len(candidate) > 0:
        return candidate
    return None


def remux_ts_to_mp4(
    input_path: Path | str,
    output_path: Path | str,
    ffmpeg_bin: Path | str | None = None,
    *,
    metadata: Any = None,
    thumbnail: bytes | Path | str | None = None,
    extra_roots: tuple[Path, ...] = (),
) -> bool:
    """
    Remux an MPEG-TS container into a faststart MP4 container using FFmpeg.

    When *metadata* or *thumbnail* is supplied, tags and attached cover art are
    written in the same single FFmpeg pass to avoid a second full-file rewrite.
    """
    ffmpeg = find_ffmpeg(ffmpeg_bin, extra_roots=extra_roots)
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg executable not found. Install FFmpeg or bundle it in third_party/.")

    in_p = Path(input_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    thumb_bytes = _extract_thumbnail_bytes(metadata, thumbnail)
    meta_fields = _extract_metadata_fields(metadata)
    temporary_thumb: Path | None = None

    try:
        cmd = [
            str(ffmpeg),
            "-y",
            "-v", "error",
            "-i", str(in_p),
        ]

        if thumb_bytes:
            suffix = ".png" if thumb_bytes.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
            with tempfile.NamedTemporaryFile(dir=out_p.parent, delete=False, suffix=suffix) as tf:
                tf.write(thumb_bytes)
                temporary_thumb = Path(tf.name)

            cmd.extend([
                "-i", str(temporary_thumb),
                "-map", "0:v:0",
                "-map", "0:a?",
                "-map", "1:v:0",
                "-c", "copy",
                "-disposition:v:1", "attached_pic",
            ])
        else:
            cmd.extend(["-c", "copy"])

        for key, val in meta_fields.items():
            cmd.extend(["-metadata", f"{key}={val}"])

        cmd.extend([
            "-movflags", "+faststart",
            str(out_p),
        ])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg stream-copy remux failed: {proc.stderr.strip()}")

        return True
    finally:
        if temporary_thumb and temporary_thumb.exists():
            try:
                temporary_thumb.unlink(missing_ok=True)
            except OSError:
                pass


def write_metadata_tags(
    path: str | Path,
    data: Any,
    ffmpeg_bin: Path | str | None = None,
    *,
    thumbnail: bytes | Path | str | None = None,
    extra_roots: tuple[Path, ...] = (),
) -> bool:
    """Write tags and optional cover thumbnail into an existing MP4 file using FFmpeg."""
    ffmpeg = find_ffmpeg(ffmpeg_bin, extra_roots=extra_roots)
    if not ffmpeg:
        logger.warning("FFmpeg not found; skipping MP4 metadata tagging.")
        return False

    target_path = Path(path).resolve()
    if not target_path.is_file():
        return False

    directory = target_path.parent
    temporary_output = directory / f".tmp_meta_{target_path.name}"
    temporary_thumb: Path | None = None

    thumb_bytes = _extract_thumbnail_bytes(data, thumbnail)
    meta_fields = _extract_metadata_fields(data)

    try:
        cmd = [
            str(ffmpeg),
            "-y",
            "-v", "error",
            "-i", str(target_path),
        ]

        if thumb_bytes:
            suffix = ".png" if thumb_bytes.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
            with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=suffix) as tf:
                tf.write(thumb_bytes)
                temporary_thumb = Path(tf.name)

            cmd.extend([
                "-i", str(temporary_thumb),
                "-map", "0:v:0",
                "-map", "0:a?",
                "-map", "1:v:0",
                "-c", "copy",
                "-disposition:v:1", "attached_pic",
            ])
        else:
            cmd.extend([
                "-map", "0",
                "-c", "copy",
            ])

        for key, val in meta_fields.items():
            cmd.extend(["-metadata", f"{key}={val}"])

        cmd.extend(["-movflags", "+faststart", str(temporary_output)])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise MetadataWriteError(f"FFmpeg tagging failed: {proc.stderr.strip()}")

        os.replace(temporary_output, target_path)
        return True

    except Exception as error:
        if isinstance(error, MetadataWriteError):
            raise
        raise MetadataWriteError(str(error)) from error

    finally:
        if temporary_output.exists():
            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                pass
        if temporary_thumb and temporary_thumb.exists():
            try:
                temporary_thumb.unlink(missing_ok=True)
            except OSError:
                pass
