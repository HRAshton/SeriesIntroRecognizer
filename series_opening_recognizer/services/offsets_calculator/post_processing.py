import logging
from typing import List, Dict

import numpy as np

from series_opening_recognizer.tp.interval import Interval

logger = logging.getLogger(__name__)


def find_most_likely_offsets(offsets_by_audio: Dict[int, List[Interval]]) -> Dict[int, Interval]:
    """
    Returns the most likely offsets for each audio file.
    """
    true_offsets_by_audio: Dict[int, Interval] = {}
    for idx, offsets in offsets_by_audio.items():
        start_offsets = [offset.start for offset in offsets]
        end_offsets = [offset.end for offset in offsets]

        start_median = np.median(start_offsets)
        end_median = np.median(end_offsets)

        true_offsets_by_audio[idx] = Interval(start_median, end_median)
        logger.debug(f'For {idx}: {start_median:.1f}, {end_median:.1f} ({end_median - start_median:.1f}s)')

    return true_offsets_by_audio
