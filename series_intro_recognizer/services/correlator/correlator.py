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


def _get_offsets_of_best_match_beat(audio1: GpuFloatArray, audio2: GpuFloatArray, cfg: Config) \
        -> tuple[GpuFloat, GpuFloat]:
    offsets_by_windows = correlation_with_async_moving_window(audio1, audio2, cfg)
    best_match = offsets_by_windows[cp.argmax(offsets_by_windows[:, 2])]

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
