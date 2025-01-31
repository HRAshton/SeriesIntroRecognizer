import logging
from typing import Iterator, Any

import cupy as cp  # type: ignore
import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.helpers.cached_iterator import iterate_with_cache
from series_intro_recognizer.services.best_offset_finder import find_best_offset
from series_intro_recognizer.services.correlator.correlator import calculate_correlation, CrossCorrelationResult
from series_intro_recognizer.services.interval_improver import improve_interval
from series_intro_recognizer.services.offsets_calculator import find_offsets
from series_intro_recognizer.tp.interval import Interval
from series_intro_recognizer.tp.tp import GpuFloatArray

logger = logging.getLogger(__name__)


def _load_to_gpu_and_normalize(audio: np.ndarray[Any, np.dtype[np.float64]]) -> GpuFloatArray:
    if audio.shape[0] == 0:
        raise ValueError('Empty audio passed.')

    gpu_audio = cp.asarray(audio, dtype=cp.float32)
    gpu_audio = gpu_audio - cp.mean(gpu_audio)
    gpu_audio = gpu_audio / cp.max(cp.abs(gpu_audio))

    return gpu_audio


def _save_corr_result(file1: int, file2: int, result: CrossCorrelationResult, cfg: Config) -> None:
    if not cfg.save_intermediate_results:
        return

    logger.info('Saving correlations for %s and %s...', file1, file2)
    with open(f'correlations/{file1}_{file2}_{result[0]:.3f}_{result[1]:.3f}.csv', 'w') as f:
        results = []
        for corr in result[2]:
            results.append(f'{corr[0]},{corr[1]}')
        f.write('\n'.join(results))


def _save_offsets_result(file1: int, file2: int, result: tuple[int, int], cfg: Config) -> None:
    if not cfg.save_intermediate_results:
        return

    logger.info('Saving offsets for %s and %s...', file1, file2)
    with open(f'offsets/{file1}.csv', 'a') as f:
        f.write(f'{file2},{result[0]},{result[1]}\n')
    with open(f'offsets/{file2}.csv', 'a') as f:
        f.write(f'{file1},{result[0]},{result[1]}\n')


def _find_offsets_for_episode(idx1: int, audio1: np.ndarray[Any, np.dtype[np.float64]],
                              idx2: int, audio2: np.ndarray[Any, np.dtype[np.float64]],
                              cfg: Config) -> tuple[Interval, Interval] | None:
    if audio1.shape[0] < cfg.min_segment_length_beats or audio2.shape[0] < cfg.min_segment_length_beats:
        logger.warning('One of the audios is shorter than %s secs: %s, %s. Skipping.',
                       cfg.min_segment_length_sec, audio1.shape[0], audio2.shape[0])
        return None

    # Step 1: Adjust the audios and calculate the correlation
    corr_result = calculate_correlation(audio1, audio2, cfg)
    if corr_result is None:
        return None
    _save_corr_result(idx1, idx2, corr_result, cfg)

    # Step 2: Find common part and its offsets
    corr_by_beats = corr_result[2][:, 1]
    offsets_result = find_offsets(corr_by_beats, cfg)
    if offsets_result is None:
        return None
    _save_offsets_result(idx1, idx2, offsets_result, cfg)

    # Step 3: Calculate the final offsets
    offset1_secs, offset2_secs, _ = corr_result
    begin1_secs = float(offset1_secs + offsets_result[0] * cfg.precision_secs)
    end1_secs = float(offset1_secs + offsets_result[1] * cfg.precision_secs)
    begin2_secs = float(offset2_secs + offsets_result[0] * cfg.precision_secs)
    end2_secs = float(offset2_secs + offsets_result[1] * cfg.precision_secs)

    # Step 4: Create intervals
    interval1 = Interval(begin1_secs, end1_secs)
    interval2 = Interval(begin2_secs, end2_secs)

    # Step 4: Improve the intervals
    audio1_duration = audio1.shape[0] / cfg.rate
    audio2_duration = audio2.shape[0] / cfg.rate
    improved_interval1 = improve_interval(interval1, audio1_duration, cfg)
    improved_interval2 = improve_interval(interval2, audio2_duration, cfg)

    logger.debug('Found offsets: %s, %s for %s and %s',
                 improved_interval1, improved_interval2, idx1, idx2)

    return improved_interval1, improved_interval2


def _find_offsets_for_episodes(audios: Iterator[np.ndarray[Any, np.dtype[np.float64]]],
                               cfg: Config) -> dict[int, list[Interval]]:
    pairs = iterate_with_cache(map(_load_to_gpu_and_normalize, audios), cfg.series_window)
    results: dict[int, list[Interval]] = {}
    for pair1, pair2 in pairs:
        idx1, audio1 = pair1
        idx2, audio2 = pair2
        logger.info(f'Processing %s and %s...', idx1, idx2)

        results.setdefault(idx1, [])
        results.setdefault(idx2, [])

        result = _find_offsets_for_episode(idx1, audio1, idx2, audio2, cfg)
        if result is None:
            continue

        results[idx1].append(result[0])
        results[idx2].append(result[1])

    cp.get_default_memory_pool().free_all_blocks()

    return results


def _find_most_likely_offsets(offsets_by_files: dict[int, list[Interval]], cfg: Config) -> dict[int, Interval]:
    """
    Returns the most likely offsets for each audio file.
    """
    true_offsets_by_files: dict[int, Interval] = {}
    for idx, offsets in offsets_by_files.items():
        true_offsets_by_files[idx] = find_best_offset(offsets, cfg)
        logger.debug('For %s: %.1f, %.1f (%.1fs)',
                     idx,
                     true_offsets_by_files[idx].start, true_offsets_by_files[idx].end,
                     true_offsets_by_files[idx].end - true_offsets_by_files[idx].start)

    return true_offsets_by_files


def recognise_from_audio_samples(audios: Iterator[np.ndarray[Any, np.dtype[np.float64]]],
                                 cfg: Config) -> list[Interval]:
    """
    Recognises series openings from a list of audio arrays.
    :param audios: list of audio arrays
    :param cfg: configuration
    :return: list of recognised intervals
    """
    offsets_by_files = _find_offsets_for_episodes(audios, cfg)
    true_offsets = _find_most_likely_offsets(offsets_by_files, cfg)
    logger.info('Results: %s', true_offsets)

    results = list(true_offsets.values())

    # Check if all indices are present in the correct order
    for i in range(len(results)):
        assert results[i] == true_offsets[i]

    return results
