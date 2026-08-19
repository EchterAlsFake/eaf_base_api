from fractions import Fraction
from types import SimpleNamespace

from base_api.base import _normalize_packet_timestamps


def test_normalize_packet_timestamps_repairs_hls_discontinuities() -> None:
    offsets: dict[int, int] = {}
    last_dts: dict[int, int] = {}
    last_durations: dict[int, int] = {}

    packets = [
        SimpleNamespace(
            stream=SimpleNamespace(index=0),
            dts=11_322_971,
            pts=11_330_478,
            duration=3_753,
            time_base=Fraction(1, 90_000),
        ),
        SimpleNamespace(
            stream=SimpleNamespace(index=0),
            dts=49_161_105,
            pts=49_168_612,
            duration=3_753,
            time_base=Fraction(1, 90_000),
        ),
        SimpleNamespace(
            stream=SimpleNamespace(index=0),
            dts=11_664_563,
            pts=11_672_070,
            duration=3_753,
            time_base=Fraction(1, 90_000),
        ),
    ]

    corrections = [
        _normalize_packet_timestamps(packet, offsets, last_dts, last_durations)
        for packet in packets
    ]

    assert corrections[0] == 0
    assert corrections[1] != 0
    assert corrections[2] != 0
    assert [packet.dts for packet in packets] == [
        11_322_971,
        11_326_724,
        11_330_477,
    ]
    assert all(packet.pts - packet.dts == 7_507 for packet in packets)
