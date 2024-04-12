from typing import Tuple

import cupy as cp

from series_opening_recognizer.configuration import RATE
from series_opening_recognizer.tp.tp import GpuFloat, GpuFloatArray


@cp.fuse()
def _compute_offsets_and_indices(offsets_diff: int, length: int) -> Tuple[float, float, int, int, int, int]:
    offset1_secs = cp.maximum(0.0, offsets_diff / RATE)
    offset2_secs = cp.maximum(0.0, -offsets_diff / RATE)

    start_idx_audio1 = cp.maximum(0, offsets_diff)
    end_idx_audio1 = start_idx_audio1 + length
    start_idx_audio2 = cp.maximum(0, -offsets_diff)
    end_idx_audio2 = start_idx_audio2 + length

    return (offset1_secs, offset2_secs,
            start_idx_audio1, end_idx_audio1,
            start_idx_audio2, end_idx_audio2)


def align_fragments(best_offset1: GpuFloat, best_offset2: GpuFloat,
                    audio1: GpuFloatArray, audio2: GpuFloatArray) -> Tuple[GpuFloatArray, GpuFloatArray, float, float]:
    offsets_diff = best_offset1 - best_offset2
    length = cp.min(cp.array([audio1.size, audio2.size])) - cp.abs(offsets_diff)

    (offset1_secs, offset2_secs,
     start_idx_audio1, end_idx_audio1,
     start_idx_audio2, end_idx_audio2) = _compute_offsets_and_indices(offsets_diff, length)

    truncated_audio1 = audio1[start_idx_audio1:end_idx_audio1]
    truncated_audio2 = audio2[start_idx_audio2:end_idx_audio2]

    return truncated_audio1, truncated_audio2, offset1_secs, offset2_secs
