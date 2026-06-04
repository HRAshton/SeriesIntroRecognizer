import math

import pytest

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.best_offset_finder import find_best_offset
from series_intro_recognizer.tp.interval import Interval

testdata: list[tuple[list[tuple[float, float]], tuple[float, float]]] = [
    (
        [(0, 10), (5, 15), (10, 20)],
        (2.5, 12.5)
    ),
    (
        [(2466, 3599), (0, 2257), (0, 2069), (0, 3597), (2485, 3349), (0, 3461)],
        (0, 3529)
    ),
    (
        [(0, 2420), (0, 2257), (0, 3064), (0, 3463), (0, 3345), (0, 3536)],
        (0, 3404)
    ),
    (
        [(0, 2600), (0, 2069), (0, 3064), (0, 1924), (0, 3289), (0, 3599)],
        (0, 3289)
    ),
    (
        [(0, 2600), (0, 2600), (0, 2600), (0, 2600), (0, 2600), (0, 2600)],
        (0, 2600)
    ),
    (
        [(0, 2600)],
        (0, 2600)
    ),
    (
        [(0.005, 2600), (0, 2600)],
        (0.005, 2600)
    ),
    (
        [(236.5, 281.0), (239.5, 360.5), (math.nan, math.nan), (147.5, 173.0), (165.0, 272.5)],
        (238.0, 281.0)
    ),
    (
        [(236.5, 281.0), (math.nan, math.nan), (math.nan, math.nan), (math.nan, math.nan)],
        (236.5, 281.0)
    ),
    (
        [],
        (math.nan, math.nan)
    ),
    (
        [(math.nan, math.nan)],
        (math.nan, math.nan)
    )
]


@pytest.mark.parametrize('offsets, expected', testdata)
def test_calculates_correct(offsets: list[tuple[float, float]], expected: tuple[float, float]) -> None:
    cfg = Config()
    intervals = [Interval(start, end) for start, end in offsets]
    expected_interval = Interval(*expected)

    result = find_best_offset(intervals, cfg)

    assert result == expected_interval


def test_too_few_clusters() -> None:
    cfg = Config(precision_secs = 1e-15)
    values = [1.0, 1.0 - 1e-6, 1.0 + 1e-6]

    result = find_best_offset([Interval(value, value) for value in values], cfg)

    assert result == Interval(1.0 + 1/2 * 1e-6, 1.0 + 1/2 * 1e-6)


def test_keeps_start_and_end_from_same_interval_cluster() -> None:
    cfg = Config()
    intervals = [
        Interval(0, 0),
        Interval(0, 10),
        Interval(0, 20),
        Interval(0, 30),
        Interval(0, 40),
        Interval(0, 50),
        Interval(0, 60),
        Interval(100, 200),
        Interval(110, 200),
        Interval(120, 200),
        Interval(130, 200),
        Interval(140, 200),
        Interval(150, 200),
    ]

    result = find_best_offset(intervals, cfg)

    assert result == Interval(0, 15)
