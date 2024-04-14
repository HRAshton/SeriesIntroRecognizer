import logging

import cupy as cp

from series_opening_recognizer.configuration import PRECISION_BEATS
from series_opening_recognizer.tp.tp import GpuFloatArray, GpuStack, GpuInt

PRECISION_BEATS_GPU = cp.array(PRECISION_BEATS)

logger = logging.getLogger(__name__)


def _create_fragments(audio: GpuFloatArray, num_frames: GpuInt) -> GpuFloatArray:
    start_indices = cp.arange(num_frames) * PRECISION_BEATS_GPU
    fragment_indices = start_indices[:, None] + cp.arange(PRECISION_BEATS_GPU)
    fragments = audio[fragment_indices]
    return fragments


def _normalize_fragments(fragments: GpuFloatArray) -> GpuFloatArray:
    mean = cp.mean(fragments, axis=1, keepdims=True)
    std = cp.std(fragments, axis=1, keepdims=True)
    return (fragments - mean) / std


def correlation_with_sync_moving_window(audio1: GpuFloatArray, audio2: GpuFloatArray) \
        -> GpuStack[GpuFloatArray, GpuFloatArray]:
    if cp.get_array_module(audio1) != cp or cp.get_array_module(audio2) != cp:
        raise ValueError("audios must be on GPU")

    if audio1.shape[0] > audio2.shape[0]:
        raise ValueError("audio2 must not be shorter than audio1")

    num_frames = audio1.shape[0] // PRECISION_BEATS_GPU
    offsets = cp.arange(num_frames) * PRECISION_BEATS_GPU

    # Create arrays for fragments
    fragments1 = _create_fragments(audio1, num_frames)
    fragments2 = _create_fragments(audio2, num_frames)

    # Normalize fragments
    normalized_fragments1 = _normalize_fragments(fragments1)
    normalized_fragments2 = _normalize_fragments(fragments2)

    # Calculate correlations and find maximum values
    max_correlations = cp.array([cp.max(cp.correlate(norm_frag1, norm_frag2, mode='full'))
                                 for norm_frag1, norm_frag2
                                 in zip(normalized_fragments1, normalized_fragments2)])

    # Combine offsets and maximum correlation values
    results = cp.stack((offsets, max_correlations), axis=-1, dtype=cp.float32)

    return results
