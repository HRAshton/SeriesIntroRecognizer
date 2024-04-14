import logging
from typing import Tuple

import cupy as cp

from series_opening_recognizer.configuration import OFFSET_SEARCHER__SEQUENTIAL_BEATS
from series_opening_recognizer.tp.tp import GpuFloatArray

logger = logging.getLogger(__name__)


def find_offsets(corr_values: GpuFloatArray) -> Tuple[int, int] or None:
    max_limit = cp.mean(corr_values) + 3 * cp.std(corr_values)
    filtered = corr_values[corr_values < max_limit]

    if cp.mean(filtered) < cp.median(filtered) * 2:
        logger.debug('Not enough correlation. Skipping.')
        return None

    if filtered.shape[0] == 0:
        logger.debug('Fragments are the same. Skipping.')
        return None

    filtered_max = cp.max(filtered)
    threshold = filtered_max / 2

    # noinspection PyTypeChecker
    begin_idx = int(cp.argmax(corr_values > threshold).get())

    end_idx = begin_idx
    bad_count = 0
    for i in range(begin_idx, len(corr_values)):
        if corr_values[i] > threshold:
            end_idx = i
        else:
            bad_count += 1
            if bad_count > OFFSET_SEARCHER__SEQUENTIAL_BEATS:
                break

    return begin_idx, end_idx
