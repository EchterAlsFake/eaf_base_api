from types import SimpleNamespace

import pytest

import base_api.base as base_module
from base_api.base import BaseCore
from base_api.modules.static_functions import (
    available_qualities,
    choose_quality_from_list,
    choose_variant,
    collect_variants,
    normalize_quality,
    normalize_qualities,
    quality_from_variant,
)


def make_variant(
    uri: str,
    resolution: tuple[int, int] | None,
    bandwidth: int,
    frame_rate: float = 30.0,
    codecs: str = "avc1.64001f,mp4a.40.2",
) -> SimpleNamespace:
    return SimpleNamespace(
        uri=uri,
        is_iframe=False,
        stream_info=SimpleNamespace(
            resolution=resolution,
            bandwidth=bandwidth,
            frame_rate=frame_rate,
            codecs=codecs,
        ),
    )


@pytest.mark.parametrize("value", [720, "720", "720p", " 720P "])
def test_normalize_quality(value: str | int) -> None:
    assert normalize_quality(value) == 720


def test_normalize_qualities_is_sorted_unique_and_tolerates_provider_labels() -> None:
    assert normalize_qualities(["720p", "auto", 240, "720", "1080p"]) == [
        240,
        720,
        1080,
    ]


def test_quality_selection_labels_and_closest_fallback() -> None:
    available = [240, 480, 1080, 2160]

    assert choose_quality_from_list(available, "worst") == 240
    assert choose_quality_from_list(available, "half") == 1080
    assert choose_quality_from_list(available, "best") == 2160
    assert choose_quality_from_list(available, "720p") == 480
    assert choose_quality_from_list([540, 900], 720) == 900
    assert choose_quality_from_list([240, 360], 144) == 240
    assert choose_quality_from_list([], "best", default_fallback="480p") == 480


def test_vertical_and_landscape_variants_have_the_same_quality() -> None:
    landscape = make_variant("landscape.m3u8", (1920, 1080), 4_000_000)
    portrait = make_variant("portrait.m3u8", (1080, 1920), 4_000_000)

    assert quality_from_variant(landscape) == 1080
    assert quality_from_variant(portrait) == 1080


def test_variants_share_discovery_and_selection_logic() -> None:
    master = SimpleNamespace(playlists=[
        make_variant("240p.m3u8", (426, 240), 300_000),
        make_variant("720p-low.m3u8", (720, 1280), 1_000_000),
        make_variant("720p-high.m3u8", (720, 1280), 2_000_000, 60.0),
        make_variant("1080p.m3u8", (1920, 1080), 4_000_000),
        make_variant("audio.m3u8", None, 128_000, codecs="mp4a.40.2"),
    ])

    variants = collect_variants(master)

    assert available_qualities(variants) == [240, 720, 1080]
    assert choose_variant(variants, "720p")["uri"] == "720p-high.m3u8"
    assert choose_variant(variants, "best")["uri"] == "1080p.m3u8"


def test_uri_is_used_when_resolution_is_missing() -> None:
    variant = make_variant("https://example.test/video/540p/index.m3u8", None, 1)
    assert quality_from_variant(variant) == 540


@pytest.mark.asyncio
async def test_base_core_uses_shared_quality_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    master = SimpleNamespace(
        is_variant=True,
        playlists=[
            make_variant("https://cdn.test/360.m3u8", (360, 640), 500_000),
            make_variant("https://cdn.test/1080.m3u8", (1080, 1920), 4_000_000),
        ],
    )
    monkeypatch.setattr(
        base_module,
        "m3u8",
        SimpleNamespace(loads=lambda _content: master),
    )
    core = BaseCore.__new__(BaseCore)
    core.logger = SimpleNamespace(debug=lambda *_args: None)
    inline_master = "#EXTM3U\n# mocked master"

    assert await core.list_available_qualities(inline_master) == [360, 1080]
    assert await core.get_m3u8_by_quality(inline_master, "720p") == (
        "https://cdn.test/1080.m3u8"
    )
