from typing import List, Tuple

import pytest

from series_opening_recognizer.services.best_offset_finder import find_best_offset
from series_opening_recognizer.tp.interval import Interval

testdata: List[Tuple[List[Tuple[int, int]], Tuple[int, int]]] = [
    (
        [(0, 10), (5, 15), (10, 20)],
        (5, 15)
    ),
    (
        [(2466, 3599), (0, 2257), (0, 2069), (0, 3597), (2485, 3349), (0, 3461)],
        (0, 3405)
    ),
    (
        [(0, 2420), (0, 2257), (0, 3064), (0, 3463), (0, 3345), (0, 3536)],
        (0, 3204.5)
    ),
    (
        [(0, 2600), (0, 2069), (0, 3064), (0, 1924), (0, 3289), (0, 3599)],
        (0, 2832)
    )
]


@pytest.mark.parametrize('offsets, expected', testdata)
def test_calculates_correct(offsets: List[Tuple[int, int]], expected: Tuple[int, int]) -> None:
    intervals = [Interval(start, end) for start, end in offsets]
    expected_interval = Interval(*expected)

    result = find_best_offset(intervals)

    assert result == expected_interval
