import cupy as cp

from series_opening_recognizer.configuration import MIN_SEGMENT_LENGTH_BEATS
from series_opening_recognizer.tp.tp import GpuFloatArray, GpuStack, GpuFloat

MIN_SEGMENT_LENGTH_BEATS_GPU = cp.array(MIN_SEGMENT_LENGTH_BEATS)


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
    if cp.get_array_module(audio1) != cp or cp.get_array_module(audio2) != cp:
        raise ValueError("audios must be on GPU")

    num_fragments = (audio1.shape[0] + MIN_SEGMENT_LENGTH_BEATS_GPU - 1) // MIN_SEGMENT_LENGTH_BEATS_GPU

    indices = cp.arange(num_fragments) * MIN_SEGMENT_LENGTH_BEATS_GPU
    fragments = audio1[indices[:, None] + cp.arange(MIN_SEGMENT_LENGTH_BEATS_GPU)]

    corr_per_fragment = cp.array([cp.correlate(audio2, fragment, mode='valid') for fragment in fragments])

    audio2_offsets = cp.argmax(corr_per_fragment, axis=1)
    corr_peaks_per_fragment = cp.max(corr_per_fragment, axis=1)

    offsets = cp.stack((indices, audio2_offsets, corr_peaks_per_fragment), axis=-1, dtype=cp.float32)

    return offsets
