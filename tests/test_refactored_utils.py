import json
import pytest
from base_api import (
    make_iterator_config,
    scrape_stream,
    stream_results,
    default_on_error,
    is_resource_gone,
    contains_resource_gone,
    parse_duration,
    parse_count,
    get_text_safe,
    get_attr_safe,
    build_m3u8_master,
    str_to_bool,
    ErrorMode,
    ErrorAction,
    NotFound,
    VideoUnavailable,
    ResourceGone,
    MediaLoadError,
    MediaLoadErrors,
    ScrapeErrorContext,
    ScrapeStage,
)


def test_make_iterator_config_defaults():
    cfg = make_iterator_config()
    assert cfg.load_specific_sources == ("html",)
    assert cfg.page_error_mode == ErrorMode.SKIP
    assert cfg._page_request_method == "GET"


def test_is_resource_gone():
    assert is_resource_gone(ResourceGone("Gone")) is True
    assert is_resource_gone(NotFound("404")) is True
    assert is_resource_gone(VideoUnavailable("Unavailable")) is True
    assert is_resource_gone(ValueError("Random")) is False

    # Nested in MediaLoadError
    wrapped = MediaLoadError("Video", "html", "http://example.com", NotFound("404"))
    assert is_resource_gone(wrapped) is True

    # Nested in MediaLoadErrors
    multi = MediaLoadErrors((ValueError("Random"), ResourceGone("Gone")))
    assert is_resource_gone(multi) is True
    assert contains_resource_gone(multi) is True


@pytest.mark.asyncio
async def test_default_on_error():
    ctx_gone = ScrapeErrorContext(
        stage=ScrapeStage.PAGE,
        url="http://example.com",
        error=NotFound("404"),
        attempt=1,
        max_attempts=3,
        page_index=0,
        item_index=None,
    )
    action = await default_on_error(ctx_gone)
    assert action == ErrorAction.SKIP

    ctx_other = ScrapeErrorContext(
        stage=ScrapeStage.PAGE,
        url="http://example.com",
        error=ValueError("Network timeout"),
        attempt=1,
        max_attempts=3,
        page_index=0,
        item_index=None,
    )
    action_other = await default_on_error(ctx_other)
    assert action_other == ErrorAction.RETRY


def test_parse_duration():
    assert parse_duration(120) == 120
    assert parse_duration("120") == 120
    assert parse_duration("PT1H2M3S") == 3723
    assert parse_duration("PT240S") == 240
    assert parse_duration("PT5M") == 300
    assert parse_duration("12:34") == 754
    assert parse_duration("1:02:03") == 3723
    assert parse_duration("59m 40s") == 3580
    assert parse_duration("24 min") == 1440
    assert parse_duration("45 seconds") == 45
    assert parse_duration("not available") is None
    assert parse_duration(None) is None


def test_parse_count():
    assert parse_count(1200) == 1200
    assert parse_count("1,234,567") == 1234567
    assert parse_count("1.2M") == 1200000
    assert parse_count("500K") == 500000
    assert parse_count("2.5B") == 2500000000
    assert parse_count(None) is None


def test_build_m3u8_master():
    defs = [
        {"format": "hls", "quality": "1080", "videoUrl": "https://cdn.example.com/1080P_4000K/video.m3u8"},
        {"format": "hls", "quality": "720", "videoUrl": "https://cdn.example.com/720P_2000K/video.m3u8"},
        {"format": "mp4", "quality": "720", "videoUrl": "https://cdn.example.com/video.mp4"},
    ]
    playlist = build_m3u8_master(defs)
    assert "#EXTM3U" in playlist
    assert "RESOLUTION=1920x1080" in playlist
    assert "RESOLUTION=1280x720" in playlist
    assert "https://cdn.example.com/1080P_4000K/video.m3u8" in playlist
    assert "video.mp4" not in playlist

    # Also accepts JSON string
    playlist_from_str = build_m3u8_master(json.dumps(defs))
    assert "#EXTM3U" in playlist_from_str
    assert "RESOLUTION=1920x1080" in playlist_from_str
