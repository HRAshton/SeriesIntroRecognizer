from typing import List

import numpy as np

from series_intro_recognizer.tp.interval import Interval


def find_best_offset(offsets: List[Interval]) -> Interval:
    """
    Returns the most likely offsets for an audio file.
    """
    start_offsets = [offset.start for offset in offsets]
    end_offsets = [offset.end for offset in offsets]

    start_median = np.median(start_offsets)
    end_median = np.median(end_offsets)

    return Interval(start_median, end_median)
