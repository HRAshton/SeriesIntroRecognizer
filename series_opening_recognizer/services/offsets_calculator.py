import logging
from typing import Tuple

import cupy as cp

from series_opening_recognizer.configuration import OFFSET_SEARCHER__SEQUENTIAL_INTERVALS
from series_opening_recognizer.tp.tp import GpuFloatArray, GpuFloat

logger = logging.getLogger(__name__)


def _find_limited_max_and_validate(corr_values: GpuFloatArray) -> GpuFloat or None:
    max_limit = cp.mean(corr_values) + 2 * cp.std(corr_values)
    filtered = corr_values[corr_values < max_limit]

    if filtered.shape[0] == 0:
        logger.debug('Fragments are the same. Skipping.')
        return None

    if cp.mean(filtered) < cp.median(filtered) * 2:
        logger.debug('Not enough correlation. Skipping.')
        return None

    return cp.max(filtered)


def find_offsets(corr_values: GpuFloatArray) -> Tuple[int, int] or None:
    limited_max = _find_limited_max_and_validate(corr_values)
    if limited_max is None:
        return None

    threshold = limited_max / 2
    bools = cp.array(corr_values > threshold)

    # Find the first peak: start
    begin_idx = cp.argmax(bools)

    # Find the first valid end after the start
    shifted_data = cp.lib.stride_tricks.sliding_window_view(
        bools,
        window_shape=(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS,))
    all_false_windows = cp.all(~shifted_data, axis=1)
    end_idx = cp.argmax(all_false_windows[begin_idx:]) + begin_idx

    return begin_idx, end_idx
