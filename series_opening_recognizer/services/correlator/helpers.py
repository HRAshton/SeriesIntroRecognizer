import logging
from typing import Tuple

import cupy as cp  # type: ignore

from series_opening_recognizer.configuration import MIN_SEGMENT_LENGTH_BEATS, PRECISION_BEATS
from series_opening_recognizer.tp.tp import GpuFloatArray, GpuStack, GpuFloat, GpuInt

MIN_SEGMENT_LENGTH_BEATS_GPU = cp.array(MIN_SEGMENT_LENGTH_BEATS)
PRECISION_BEATS_GPU = cp.array(PRECISION_BEATS)

logger = logging.getLogger(__name__)


def correlation_with_async_moving_window(audio1: GpuFloatArray,
                                         audio2: GpuFloatArray) -> GpuStack[GpuFloat, GpuFloat, GpuFloat]:
    """
    Splits audio1 into fragments of size MIN_SEGMENT_LENGTH_BEATS_GPU and calculates correlation with audio2.
    Returns a list of tuples (offset1, offset2, corr) containing data about the offset of the most similar fragment
    of audio2 for each fragment of audio1.
    :param audio1: ndarray with audio
    :param audio2: ndarray with audio
    :return: list of tuples (offset1, offset2, corr),
                where offset1 - offset in audio1,
                      offset2 - offset in audio2,
                      corr - correlation coefficient
    """
    if cp.get_array_module(audio1) != cp:
        raise ValueError("audio1 must be on GPU")

    num_fragments = (audio1.shape[0] + MIN_SEGMENT_LENGTH_BEATS_GPU - 1) // MIN_SEGMENT_LENGTH_BEATS_GPU

    indices = cp.arange(num_fragments - 1) * MIN_SEGMENT_LENGTH_BEATS_GPU
    fragments = audio1[indices[:, None] + cp.arange(MIN_SEGMENT_LENGTH_BEATS_GPU)]

    corr_per_fragment = cp.array([cp.correlate(audio2, fragment, mode='valid') for fragment in fragments])

    audio2_offsets = cp.argmax(corr_per_fragment, axis=1)
    corr_peaks_per_fragment = cp.max(corr_per_fragment, axis=1)

    offsets = cp.stack((indices, audio2_offsets, corr_peaks_per_fragment), axis=-1)

    return offsets


def create_fragments(audio: GpuFloatArray, num_frames: GpuInt, precision_beats: GpuInt) -> GpuFloatArray:
    start_indices = cp.arange(num_frames) * precision_beats
    fragment_indices = start_indices[:, None] + cp.arange(precision_beats)
    fragments = audio[fragment_indices]
    return fragments


def normalize_fragments(fragments: GpuFloatArray) -> GpuFloatArray:
    mean = cp.mean(fragments, axis=1, keepdims=True)
    std = cp.std(fragments, axis=1, keepdims=True)
    return (fragments - mean) / std


def correlation_with_sync_moving_window(audio1: GpuFloatArray, audio2: GpuFloatArray) \
        -> GpuStack[GpuFloatArray, GpuFloatArray]:
    if audio1.shape[0] > audio2.shape[0]:
        raise ValueError("audio2 must not be shorter than audio1")

    num_frames = audio1.shape[0] // PRECISION_BEATS_GPU
    offsets = cp.arange(num_frames) * PRECISION_BEATS_GPU

    # Create arrays for fragments
    fragments1 = create_fragments(audio1, num_frames, PRECISION_BEATS_GPU)
    fragments2 = create_fragments(audio2, num_frames, PRECISION_BEATS_GPU)

    # Normalize fragments
    normalized_fragments1 = normalize_fragments(fragments1)
    normalized_fragments2 = normalize_fragments(fragments2)

    # Calculate correlations and find maximum values
    max_correlations = cp.array([cp.max(cp.correlate(norm_frag1, norm_frag2, mode='full'))
                                 for norm_frag1, norm_frag2
                                 in zip(normalized_fragments1, normalized_fragments2)])

    # Combine offsets and maximum correlation values
    results = cp.stack((offsets, max_correlations), axis=-1)

    return results
