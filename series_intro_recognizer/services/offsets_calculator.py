import logging

import cupy as cp  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.tp.tp import GpuFloatArray, GpuFloat

logger = logging.getLogger(__name__)


def _get_threshold(corr_values: GpuFloatArray) -> GpuFloat | None:
    max_limit = cp.mean(corr_values) + 2 * cp.std(corr_values)
    filtered = corr_values[corr_values < max_limit]

    if filtered.shape[0] == 0:
        logger.warning('Fragments are the same. Skipping. Try to increase the samples length.')
        return None

    return cp.max(filtered) / 2


def _longest_sequence_with_gaps(arr: GpuFloatArray, max_gap_length: int) -> tuple[int, int]:
    kernel = cp.ElementwiseKernel(
        in_params='raw bool arr, int32 max_gap_length',
        out_params='int32 max_start, int32 max_end',
        operation='''
            int n = arr.size();
            int current_start = -1;
            int current_end = -1;
            int longest_start = -1;
            int longest_end = -1;
            int gap_length = 0;

            for (int i = 0; i < n; i++) {
                if (arr[i]) {
                    if (current_start == -1) {
                        current_start = i;
                    }
                    current_end = i;
                    gap_length = 0;
                } else {
                    if (current_start != -1) {
                        gap_length++;
                        if (gap_length > max_gap_length) {
                            if ((longest_start == -1) || (current_end - current_start > longest_end - longest_start)) {
                                longest_start = current_start;
                                longest_end = current_end;
                            }
                            current_start = -1;
                            current_end = -1;
                            gap_length = 0;
                        }
                    }
                }
            }

            if ((current_start != -1) && (current_end - current_start > longest_end - longest_start)) {
                longest_start = current_start;
                longest_end = current_end;
            }

            max_start = longest_start;
            max_end = longest_end;
        ''',
        name='longest_sequence_with_gaps_gpu'
    )

    max_start = cp.zeros(1, dtype=cp.int32)
    max_end = cp.zeros(1, dtype=cp.int32)

    kernel(arr, max_gap_length, max_start, max_end)

    return int(max_start[0]), int(max_end[0])


def find_offsets(corr_values: GpuFloatArray, cfg: Config) -> tuple[int, int] | None:
    threshold = _get_threshold(corr_values)
    if threshold is None:
        return None

    bools = cp.asarray(corr_values > threshold)
    start, end = _longest_sequence_with_gaps(bools, cfg.offset_searcher_sequential_intervals)

    # Try to include the next element, because the end is exclusive
    # However, it would be incorrect if the end is at the last element,
    # so we need to check if it is the case.
    return start, min(end + 1, corr_values.size)
