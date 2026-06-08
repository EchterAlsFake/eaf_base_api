from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

