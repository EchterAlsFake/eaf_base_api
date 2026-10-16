from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from base_api import (
    BaseCore,
    MetadataWriteError,
    find_ffmpeg,
    has_ffmpeg,
    remux_ts_to_mp4,
    write_metadata_tags,
)

SAMPLE_TS = Path("/home/asuna/PycharmProjects/Porn_Fetch/src/tests/fixtures/ffmpeg/sample.ts")
COVER_JPG = Path("/home/asuna/PycharmProjects/Porn_Fetch/src/tests/fixtures/ffmpeg/cover.jpg")
COVER_PNG = Path("/home/asuna/PycharmProjects/Porn_Fetch/src/tests/fixtures/ffmpeg/cover.png")


def _get_test_ffmpeg() -> Path | None:
    pf_ffmpeg = Path("/home/asuna/PycharmProjects/Porn_Fetch/third_party/ffmpeg")
    if pf_ffmpeg.is_file() and os.access(pf_ffmpeg, os.X_OK):
        return pf_ffmpeg
    return find_ffmpeg()


def test_find_ffmpeg_respects_explicit_and_env_override(tmp_path: Path) -> None:
    fake_bin = tmp_path / "ffmpeg"
    fake_bin.write_bytes(b"#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    assert find_ffmpeg(fake_bin) == fake_bin.resolve()
    assert has_ffmpeg(fake_bin) is True

    with patch.dict(os.environ, {"PORN_FETCH_FFMPEG_BINARY": str(fake_bin)}):
        assert find_ffmpeg() == fake_bin.resolve()


@pytest.mark.skipif(
    not SAMPLE_TS.is_file() or _get_test_ffmpeg() is None,
    reason="Requires sample.ts fixture and FFmpeg executable",
)
def test_single_pass_remux_ts_to_mp4_with_metadata_and_cover(tmp_path: Path) -> None:
    ffmpeg_bin = _get_test_ffmpeg()
    assert ffmpeg_bin is not None

    output_mp4 = tmp_path / "remuxed_single_pass.mp4"
    meta = SimpleNamespace(
        title="Single Pass Remux Title",
        author="Single Pass Artist",
        publish_date=datetime(2026, 10, 16, tzinfo=timezone.utc),
        thumbnail_data=COVER_JPG.read_bytes(),
    )

    assert remux_ts_to_mp4(SAMPLE_TS, output_mp4, ffmpeg_bin=ffmpeg_bin, metadata=meta) is True
    assert output_mp4.is_file()

    raw = output_mp4.read_bytes()
    assert raw.index(b"moov") < raw.index(b"mdat")

    probe = subprocess.run(
        [str(ffmpeg_bin), "-hide_banner", "-i", str(output_mp4)],
        capture_output=True,
        text=True,
    )
    assert "Video: h264" in probe.stderr
    assert "Audio: aac" in probe.stderr
    assert "Single Pass Remux Title" in probe.stderr
    assert "Single Pass Artist" in probe.stderr
    assert "attached pic" in probe.stderr


@pytest.mark.skipif(
    not SAMPLE_TS.is_file() or _get_test_ffmpeg() is None,
    reason="Requires sample.ts fixture and FFmpeg executable",
)
def test_basecore_convert_ts_to_mp4_single_pass_and_fallback(tmp_path: Path) -> None:
    ffmpeg_bin = _get_test_ffmpeg()
    assert ffmpeg_bin is not None

    core = BaseCore()
    core.configuration.ffmpeg_path = str(ffmpeg_bin)

    out_ok = tmp_path / "core_ok.mp4"
    progress_calls: list[tuple[int, int]] = []
    meta = {"title": "Core Remux Title", "author": "Core Author"}

    wrote_meta = core._convert_ts_to_mp4(
        str(SAMPLE_TS),
        str(out_ok),
        callback=lambda pos, tot: progress_calls.append((pos, tot)),
        metadata=meta,
        thumbnail=COVER_PNG.read_bytes(),
    )
    assert wrote_meta is True
    assert out_ok.is_file()
    assert len(progress_calls) == 1
    assert progress_calls[0][0] == progress_calls[0][1] > 0

    # Corrupt thumbnail bytes should gracefully fall back to plain remux without losing the video
    out_fallback = tmp_path / "core_fallback.mp4"
    wrote_meta_fallback = core._convert_ts_to_mp4(
        str(SAMPLE_TS),
        str(out_fallback),
        metadata=meta,
        thumbnail=b"not-a-valid-image",
    )
    assert wrote_meta_fallback is False
    assert out_fallback.is_file()
    assert out_fallback.stat().st_size > 0


@pytest.mark.skipif(
    not SAMPLE_TS.is_file() or _get_test_ffmpeg() is None,
    reason="Requires sample.ts fixture and FFmpeg executable",
)
def test_write_metadata_tags_on_existing_mp4(tmp_path: Path) -> None:
    ffmpeg_bin = _get_test_ffmpeg()
    assert ffmpeg_bin is not None

    mp4_path = tmp_path / "existing.mp4"
    remux_ts_to_mp4(SAMPLE_TS, mp4_path, ffmpeg_bin=ffmpeg_bin)

    meta = SimpleNamespace(
        title="Retagged Title",
        author="Retagged Artist",
        thumbnail_data=COVER_PNG.read_bytes(),
    )
    assert write_metadata_tags(mp4_path, meta, ffmpeg_bin=ffmpeg_bin) is True

    probe = subprocess.run(
        [str(ffmpeg_bin), "-hide_banner", "-i", str(mp4_path)],
        capture_output=True,
        text=True,
    )
    assert "Retagged Title" in probe.stderr
    assert "Retagged Artist" in probe.stderr
    assert "attached pic" in probe.stderr

    with pytest.raises(MetadataWriteError):
        write_metadata_tags(mp4_path, meta, ffmpeg_bin=ffmpeg_bin, thumbnail=b"corrupt-image")
