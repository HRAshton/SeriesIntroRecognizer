import cupy as cp  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.tp.tp import GpuFloat, GpuFloatArray, GpuInt


@cp.fuse()  # type: ignore
def _compute_offsets_and_indices(offsets_diff: int, length: int, rate: int) \
        -> tuple[GpuFloat, GpuFloat, GpuInt, GpuInt, GpuInt, GpuInt]:
    offset1_secs = cp.maximum(0.0, offsets_diff / rate)
    offset2_secs = cp.maximum(0.0, -offsets_diff / rate)

    start_idx_audio1 = cp.maximum(0, offsets_diff)
    end_idx_audio1 = start_idx_audio1 + length
    start_idx_audio2 = cp.maximum(0, -offsets_diff)
    end_idx_audio2 = start_idx_audio2 + length

    return (offset1_secs, offset2_secs,
            start_idx_audio1, end_idx_audio1,
            start_idx_audio2, end_idx_audio2)


def align_fragments(best_offset1: GpuFloat, best_offset2: GpuFloat,
                    audio1: GpuFloatArray, audio2: GpuFloatArray,
                    cfg: Config) -> tuple[GpuFloatArray, GpuFloatArray, GpuFloat, GpuFloat]:
    """
    Aligns two audio fragments based on the best offsets found by the correlator.
    Returns two audio fragments with the same duration, where the best_offsets are placed at the same point.
    For example, if there are two audios with duration 4s and 13s, and the best offsets are 1s and 3s,
    the resulting audios will have duration 4s.
    The 1st audio will be truncated from 0s to 4s, and the 2nd audio will be truncated from 3s to 7s.
    :param best_offset1: Offset of the common part of the audio fragments in the first audio fragment.
    :param best_offset2: Offset of the common part of the audio fragments in the second audio fragment.
    :param audio1:
    :param audio2:
    :param cfg: Configuration.
    :return: Tuple of truncated audio fragments and the offsets in seconds.
    """
    offsets_diff = best_offset1 - best_offset2
    length = cp.min(cp.array([audio1.size, audio2.size])) - cp.abs(offsets_diff)

    (offset1_secs, offset2_secs,
     start_idx_audio1, end_idx_audio1,
     start_idx_audio2, end_idx_audio2) = _compute_offsets_and_indices(offsets_diff, length, cfg.rate)

    truncated_audio1 = audio1[start_idx_audio1:end_idx_audio1]
    truncated_audio2 = audio2[start_idx_audio2:end_idx_audio2]

    return truncated_audio1, truncated_audio2, offset1_secs, offset2_secs
