import logging
from typing import Iterable, List, Dict, Tuple

import cupy as cp
import numpy as np

from series_opening_recognizer.configuration import (MIN_SEGMENT_LENGTH_BEATS, MIN_SEGMENT_LENGTH_SEC, SERIES_WINDOW,
                                                     DEBUG_SAVE_INTERMEDIATE_RESULTS, PRECISION_SECS)
from series_opening_recognizer.helpers.cached_iterator import iterate_with_cache
from series_opening_recognizer.services.best_offset_finder import find_best_offset
from series_opening_recognizer.services.correlator.correlator import calculate_correlation, CrossCorrelationResult
from series_opening_recognizer.services.offsets_calculator import find_offsets
from series_opening_recognizer.tp.interval import Interval
from series_opening_recognizer.tp.tp import GpuFloatArray

logger = logging.getLogger(__name__)


@cp.fuse()
def _load_to_gpu_and_normalize(audio: np.ndarray) -> GpuFloatArray:
    gpu_audio = cp.asarray(audio, dtype=cp.float32)
    gpu_audio = gpu_audio - cp.mean(gpu_audio)
    gpu_audio = gpu_audio / cp.max(cp.abs(gpu_audio))

    return gpu_audio


def _save_corr_result(file1: int, file2: int, result: CrossCorrelationResult) -> None:
    if not DEBUG_SAVE_INTERMEDIATE_RESULTS:
        return

    logger.info(f'Saving correlations for {file1} and {file2}...')
    with open(f'correlations/{file1}_{file2}_{result[0]:.3f}_{result[1]:.3f}.csv', 'w') as f:
        results = []
        for corr in result[2]:
            results.append(f'{corr[0]},{corr[1]}')
        f.write('\n'.join(results))


def _save_offsets_result(file1: int, file2: int, result: Tuple[int, int]) -> None:
    if not DEBUG_SAVE_INTERMEDIATE_RESULTS:
        return

    logger.info(f'Saving offsets for {file1} and {file2}...')
    with open(f'offsets/{file1}.csv', 'a') as f:
        f.write(f'{file2},{result[0]},{result[1]}\n')
    with open(f'offsets/{file2}.csv', 'a') as f:
        f.write(f'{file1},{result[0]},{result[1]}\n')


def _find_offsets_for_episodes(audios: Iterable[np.ndarray]) -> Dict[int, List[Interval]]:
    pairs = iterate_with_cache(map(_load_to_gpu_and_normalize, audios), SERIES_WINDOW)
    results: Dict[int, List[Interval]] = {}
    for pair1, pair2 in pairs:
        idx1, audio1 = pair1
        idx2, audio2 = pair2
        logger.info(f'Processing {idx1} and {idx2}...')

        if audio1.shape[0] < MIN_SEGMENT_LENGTH_BEATS or audio2.shape[0] < MIN_SEGMENT_LENGTH_BEATS:
            logger.warning('One of the audios is shorter than %s secs: %s, %s. Skipping.',
                           MIN_SEGMENT_LENGTH_SEC, audio1.shape[0], audio2.shape[0])
            continue

        corr_result = calculate_correlation(audio1, audio2)
        if corr_result is None:
            continue
        _save_corr_result(idx1, idx2, corr_result)

        corr_by_beats = corr_result[2][:, 1]
        offsets_result = find_offsets(corr_by_beats)
        if offsets_result is None:
            continue
        _save_offsets_result(idx1, idx2, offsets_result)

        offset1_secs, offset2_secs, _ = corr_result
        begin1_start_secs = float(offset1_secs + offsets_result[0] * PRECISION_SECS)
        end1_start_secs = float(offset1_secs + offsets_result[1] * PRECISION_SECS)
        begin2_start_secs = float(offset2_secs + offsets_result[0] * PRECISION_SECS)
        end2_start_secs = float(offset2_secs + offsets_result[1] * PRECISION_SECS)

        logger.debug('Found offsets: %s, %s, %s, %s for %s and %s',
                     begin1_start_secs, end1_start_secs, begin2_start_secs, end2_start_secs, idx1, idx2)
        results.setdefault(idx1, []).append(Interval(begin1_start_secs, end1_start_secs))
        results.setdefault(idx2, []).append(Interval(begin2_start_secs, end2_start_secs))

    cp.get_default_memory_pool().free_all_blocks()

    return results


def find_most_likely_offsets(offsets_by_files: Dict[int, List[Interval]]) -> Dict[int, Interval]:
    """
    Returns the most likely offsets for each audio file.
    """
    true_offsets_by_files: Dict[int, Interval] = {}
    for idx, offsets in offsets_by_files.items():
        true_offsets_by_files[idx] = find_best_offset(offsets)
        logger.debug(f'For {idx}: {true_offsets_by_files[idx].start:.1f}, '
                     f'{true_offsets_by_files[idx].end:.1f} '
                     f'({true_offsets_by_files[idx].end - true_offsets_by_files[idx].start:.1f}s)')

    return true_offsets_by_files


def recognise_from_audio_samples(audios: Iterable[np.ndarray]) -> Dict[int, Interval]:
    """
    Recognises series openings from a list of audio arrays.
    :param audios: list of audio arrays
    :return: list of tuples with indices of the audios and CrossCorrelationResult
    """
    offsets_by_files = _find_offsets_for_episodes(audios)
    true_offsets = find_most_likely_offsets(offsets_by_files)
    logger.info('Results: %s', true_offsets)

    return true_offsets
