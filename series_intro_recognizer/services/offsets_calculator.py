import logging

import cupy as cp  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.helpers.telemetry import telemetry
from series_intro_recognizer.tp.tp import GpuFloatArray, GpuFloat

logger = logging.getLogger(__name__)

_LONGEST_SEQUENCE_WITH_GAPS_KERNEL = cp.RawKernel(
    code=r'''
    extern "C" __global__
    void longest_sequence_with_gaps(
        const bool* arr,
        const int n,
        const int max_gap_length,
        const int min_continuous_length,
        const int max_ignored_trailing_length,
        int* out_start,
        int* out_end
    ) {
        if (blockIdx.x != 0 || threadIdx.x != 0) {
            return;
        }

        int current_start = -1;
        int current_end = -1;
        int current_positive_run = 0;
        int current_max_positive_run = 0;
        int current_max_positive_run_before_last_run = 0;
        int current_end_before_last_run = -1;
        int last_positive_run_length = 0;
        int longest_start = -1;
        int longest_end = -1;
        int gap_length = 0;

        for (int i = 0; i < n; i++) {
            if (arr[i]) {
                if (current_start == -1) {
                    current_start = i;
                }
                if (gap_length > 0) {
                    current_end_before_last_run = current_end;
                    current_max_positive_run_before_last_run = current_max_positive_run;
                    current_positive_run = 0;
                }
                current_end = i;
                current_positive_run++;
                last_positive_run_length = current_positive_run;
                if (current_positive_run > current_max_positive_run) {
                    current_max_positive_run = current_positive_run;
                }
                gap_length = 0;
            } else if (current_start != -1) {
                current_positive_run = 0;
                gap_length++;
                if (gap_length > max_gap_length) {
                    int candidate_end = current_end;
                    int candidate_max_positive_run = current_max_positive_run;
                    if (
                        max_ignored_trailing_length > 0
                        && current_end_before_last_run >= current_start
                        && last_positive_run_length <= max_ignored_trailing_length
                    ) {
                        candidate_end = current_end_before_last_run;
                        candidate_max_positive_run = current_max_positive_run_before_last_run;
                    }
                    if (
                        candidate_max_positive_run >= min_continuous_length
                        && ((longest_start == -1) || (candidate_end - current_start > longest_end - longest_start))
                    ) {
                        longest_start = current_start;
                        longest_end = candidate_end;
                    }
                    current_start = -1;
                    current_end = -1;
                    current_positive_run = 0;
                    current_max_positive_run = 0;
                    current_max_positive_run_before_last_run = 0;
                    current_end_before_last_run = -1;
                    last_positive_run_length = 0;
                    gap_length = 0;
                }
            }
        }

        int candidate_end = current_end;
        int candidate_max_positive_run = current_max_positive_run;
        if (
            max_ignored_trailing_length > 0
            && current_end_before_last_run >= current_start
            && last_positive_run_length <= max_ignored_trailing_length
        ) {
            candidate_end = current_end_before_last_run;
            candidate_max_positive_run = current_max_positive_run_before_last_run;
        }
        if (
            (current_start != -1)
            && candidate_max_positive_run >= min_continuous_length
            && (candidate_end - current_start > longest_end - longest_start)
        ) {
            longest_start = current_start;
            longest_end = candidate_end;
        }

        out_start[0] = longest_start;
        out_end[0] = longest_end;
    }
    ''',
    name='longest_sequence_with_gaps',
)


def _get_threshold(corr_values: GpuFloatArray) -> GpuFloat | None:
    max_limit = cp.mean(corr_values) + 2 * cp.std(corr_values)
    filtered = corr_values[corr_values < max_limit]

    if filtered.shape[0] == 0:
        logger.debug('All correlations are above the maximum limit')
        return None

    return cp.max(filtered) / 2


def _longest_sequence_with_gaps(
        arr: GpuFloatArray,
        max_gap_length: int,
        min_continuous_length: int,
        max_ignored_trailing_length: int) -> tuple[int, int]:
    n = int(arr.size)
    if n == 0:
        return -1, -1

    out_start = cp.empty(1, dtype=cp.int32)
    out_end = cp.empty(1, dtype=cp.int32)
    _LONGEST_SEQUENCE_WITH_GAPS_KERNEL(
        (1,),
        (1,),
        (arr, n, max_gap_length, min_continuous_length, max_ignored_trailing_length, out_start, out_end),
    )

    return int(out_start[0]), int(out_end[0])


def _find_offsets(corr_values: GpuFloatArray, cfg: Config) -> tuple[int, int] | None:
    if corr_values.size == 0:
        logger.warning('No correlation values provided. Skipping.')
        return None

    too_close = cp.allclose(corr_values, corr_values[0], rtol=cfg.offset_searcher_similarity_too_close_coeff)
    if too_close:
        logger.warning('The found correlations are too close to each other. Skipping.')
        return None

    threshold = _get_threshold(corr_values)
    if threshold is None:
        logger.warning('Could not determine a valid threshold for correlations. Skipping.')
        return None

    bools = cp.asarray(corr_values > threshold)
    start, end = _longest_sequence_with_gaps(
        bools,
        cfg.offset_calculator_max_gap_intervals,
        cfg.offset_calculator_min_continuous_positive_intervals,
        cfg.offset_calculator_max_ignored_trailing_positive_intervals,
    )

    if start < 0:
        logger.warning('No correlation sequence contains enough continuous positive values. Skipping.')
        return None

    # Try to include the next element, because the end is exclusive
    # However, it would be incorrect if the end is at the last element,
    # so we need to check if it is the case.
    return start, min(end + 1, corr_values.size)


def find_offsets(corr_values: GpuFloatArray, cfg: Config) -> tuple[int, int] | None:
    with telemetry.measure('find_offsets'):
        return _find_offsets(corr_values, cfg)
