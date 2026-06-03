import math

import cupy as cp  # type: ignore
import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.correlator.correlator import calculate_correlation, _get_best_offsets_pair


def _assert_cupy_row_equal(actual: cp.ndarray, expected: cp.ndarray) -> None:
    assert cp.allclose(actual, expected), f'{actual.get()} != {expected.get()}'


def test_get_best_offsets_pair_always_choose_best_score() -> None:
    cfg = Config(correlator_always_choose_best_score=True)
    offsets_by_windows = cp.asarray([
        [0, 10, 0.8],
        [100, 110, 0.7],
        [200, 500, 0.9],
    ], dtype=cp.float32)

    result = _get_best_offsets_pair(offsets_by_windows, cfg)

    _assert_cupy_row_equal(result, offsets_by_windows[2])


def test_get_best_offsets_pair_prefers_largest_lag_cluster() -> None:
    cfg = Config()
    offsets_by_windows = cp.asarray([
        [0, 10, 0.5],
        [100, 110, 0.7],
        [200, 500, 0.9],
    ], dtype=cp.float32)

    result = _get_best_offsets_pair(offsets_by_windows, cfg)

    _assert_cupy_row_equal(result, offsets_by_windows[1])


def test_get_best_offsets_pair_falls_back_to_best_score_when_no_lag_cluster() -> None:
    cfg = Config()
    offsets_by_windows = cp.asarray([
        [0, 10, 0.5],
        [100, 120, 0.7],
        [200, 500, 0.9],
    ], dtype=cp.float32)

    result = _get_best_offsets_pair(offsets_by_windows, cfg)

    _assert_cupy_row_equal(result, offsets_by_windows[2])


def test_get_best_offsets_pair_uses_best_score_across_tied_lag_clusters() -> None:
    cfg = Config()
    offsets_by_windows = cp.asarray([
        [0, 10, 0.5],
        [100, 110, 0.7],
        [200, 220, 0.8],
        [300, 320, 0.6],
    ], dtype=cp.float32)

    result = _get_best_offsets_pair(offsets_by_windows, cfg)

    _assert_cupy_row_equal(result, offsets_by_windows[2])


def test_integration_returns_none_when_no_correlation() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(1000)
    audio2 = cp.random.default_rng(0).random(1000)

    result = calculate_correlation(audio1, audio2, cfg)

    assert result is None


def test_integration_calculates_correctly() -> None:
    cfg = Config()
    offset1 = int(4.2 * cfg.rate)
    offset2 = int(7.3 * cfg.rate)
    common_part_size = int(2.2 * cfg.rate)

    audio1 = cp.random.default_rng(0).random(cfg.rate * 30)
    audio2 = cp.random.default_rng(1).random(cfg.rate * 45)
    common_part = cp.random.default_rng(2).random(common_part_size)
    audio1[offset1:offset1 + common_part.size] = common_part
    audio2[offset2:offset2 + common_part.size] = common_part

    result = calculate_correlation(audio1, audio2, cfg)

    precision_beats_multiplier = cfg.rate * cfg.precision_secs

    assert result is not None, 'Result should not be None'
    assert cp.isclose(result[0], cp.array(0)), 'Audio 1 should have a correct offset'
    assert cp.isclose(result[1], cp.array(3.1)), 'Audio 2 should have a correct offset'
    assert result[2].shape[0] == int((30 - (7.3 - 4.2)) / cfg.precision_secs), \
        'Correlation should have correct size'
    assert result[2].shape[1] == 2, 'Correlation should have 2 columns'

    corr = result[2].get()
    observed_beat_values = corr[:, 0]
    expected_beat_values = cp.arange(0, observed_beat_values.size, 1) * precision_beats_multiplier
    assert np.allclose(observed_beat_values, expected_beat_values), 'Correlation should have correct indices'

    values = corr[:, 1]
    mean = np.mean(values)

    idx_start = math.floor(offset1 / precision_beats_multiplier)
    idx_end = math.ceil((offset1 + common_part_size) / precision_beats_multiplier)
    peak_mask = np.zeros(values.shape[0], dtype=bool)
    peak_mask[idx_start:idx_end] = True
    assert np.all(values[peak_mask] > mean), 'Peak values should be higher than mean'

    non_peak_mask = ~peak_mask
    assert np.all(values[non_peak_mask] < mean), 'Non-peak values should be lower than mean'
