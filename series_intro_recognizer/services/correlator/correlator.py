import logging
from typing import Annotated

import cupy as cp  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.correlator.async_correlator import correlation_with_async_moving_window
from series_intro_recognizer.services.correlator.fragments_normalizer import align_fragments
from series_intro_recognizer.services.correlator.sync_correlator import correlation_with_sync_moving_window
from series_intro_recognizer.tp.tp import GpuFloatArray, GpuStack, GpuFloat

CrossCorrelationResult = Annotated[
    tuple[GpuFloat, GpuFloat, GpuStack[GpuFloatArray, GpuFloatArray, None]],
    'CrossCorrelationResult']

logger = logging.getLogger(__name__)


def _get_best_offsets_pair(
    offsets_by_windows: GpuStack[GpuFloat, GpuFloat, GpuFloat],
    cfg: Config,
) -> GpuFloatArray:
    """
    Select best async match.

    Prefer lag clusters:
        lag = offset2 - offset1

    Rank by:
        1. number of candidates with similar lag
        2. best score inside that lag cluster
    """
    if cfg.correlator_always_choose_best_score:
        return offsets_by_windows[cp.argmax(offsets_by_windows[:, 2])]

    lags = offsets_by_windows[:, 1] - offsets_by_windows[:, 0]
    scores = offsets_by_windows[:, 2]

    exact_match = _get_best_offsets_pair_by_cluster(offsets_by_windows, lags, scores)
    if exact_match is not None:
        return exact_match

    tolerance = cfg.correlator_lag_tolerance_beats
    if tolerance > 0:
        lag_buckets = cp.rint(lags / tolerance)
        approximate_match = _get_best_offsets_pair_by_cluster(offsets_by_windows, lag_buckets, scores)
        if approximate_match is not None:
            return approximate_match

    return offsets_by_windows[cp.argmax(scores)]


def _get_best_offsets_pair_by_cluster(
    offsets_by_windows: GpuStack[GpuFloat, GpuFloat, GpuFloat],
    cluster_keys: GpuFloatArray,
    scores: GpuFloatArray,
) -> GpuFloatArray | None:
    unique_lags, inverse, counts = cp.unique(
        cluster_keys,
        return_inverse=True,
        return_counts=True,
    )

    best_count = cp.max(counts)

    if int(best_count) <= 1:
        return None

    # lag values belonging to the largest cluster
    best_lag_ids = cp.where(counts == best_count)[0]
    candidate_mask = cp.isin(inverse, best_lag_ids)
    candidate_scores = cp.where(candidate_mask, scores, -cp.inf)

    return offsets_by_windows[cp.argmax(candidate_scores)]


def _get_offsets_of_best_match_beat(audio1: GpuFloatArray, audio2: GpuFloatArray, cfg: Config) \
        -> tuple[GpuFloat, GpuFloat]:
    offsets_by_windows = correlation_with_async_moving_window(audio1, audio2, cfg)
    best_match = _get_best_offsets_pair(offsets_by_windows, cfg)

    return best_match[0], best_match[1]


def calculate_correlation(audio1: GpuFloatArray, audio2: GpuFloatArray, cfg: Config) -> CrossCorrelationResult | None:
    """
    Aligns two audios and calculates correlation.
    :param audio1: audio1
    :param audio2: audio2
    :param cfg: Config
    :return: CrossCorrelationResult or None
    """
    best_offset1, best_offset2 = _get_offsets_of_best_match_beat(audio1, audio2, cfg)

    truncated_audio1, truncated_audio2, offset1_secs, offset2_secs = \
        align_fragments(best_offset1, best_offset2, audio1, audio2, cfg)
    if (truncated_audio1.shape[0] == 0
            or truncated_audio2.shape[0] == 0
            or truncated_audio1.shape[0] != truncated_audio2.shape[0]):
        # I believe this is not possible, but just in case
        logger.error('Truncated audios have different lengths: %s, %s. Skipping.',
                     truncated_audio1.shape[0], truncated_audio2.shape[0])
        return None

    corr_by_beats = correlation_with_sync_moving_window(truncated_audio1, truncated_audio2, cfg)

    return offset1_secs, offset2_secs, corr_by_beats
