import cupy as cp
import numpy as np

from series_opening_recognizer.configuration import PRECISION_BEATS
from series_opening_recognizer.services.correlator.sync_correlator import correlation_with_sync_moving_window


def test_return_type():
    audio1 = cp.random.default_rng(0).random(PRECISION_BEATS * 10)
    audio2 = cp.random.default_rng(1).random(PRECISION_BEATS * 10)

    result = correlation_with_sync_moving_window(audio1, audio2)

    assert isinstance(result, cp.ndarray)
    assert isinstance(result[0, 0], cp.ndarray)
    assert isinstance(result[0, 1], cp.ndarray)


def test_return_cupy_data():
    audio1 = cp.random.default_rng(0).random(PRECISION_BEATS * 10)
    audio2 = cp.random.default_rng(1).random(PRECISION_BEATS * 10)

    result = correlation_with_sync_moving_window(audio1, audio2)

    assert cp.get_array_module(result) == cp
    assert cp.get_array_module(result[0, 0]) == cp
    assert cp.get_array_module(result[0, 1]) == cp


def test_returns_array():
    audio1 = cp.random.default_rng(0).random(PRECISION_BEATS * 10)
    audio2 = cp.random.default_rng(1).random(PRECISION_BEATS * 10)

    result = correlation_with_sync_moving_window(audio1, audio2)

    assert result.shape[1] == 2


def test_throws_error_on_cpu_array():
    audio1 = np.random.default_rng(0).random(PRECISION_BEATS * 10)
    audio2 = np.random.default_rng(1).random(PRECISION_BEATS * 10)

    try:
        correlation_with_sync_moving_window(audio1, audio2)
        assert False
    except ValueError:
        assert True


def test_throws_error_on_audio1_longer_than_audio2():
    audio1 = cp.random.default_rng(0).random(PRECISION_BEATS * 20)
    audio2 = cp.random.default_rng(1).random(PRECISION_BEATS * 10)

    try:
        correlation_with_sync_moving_window(audio1, audio2)
        assert False
    except ValueError:
        assert True


def test_returns_empty_array_on_empty_input():
    audio1 = cp.array([])
    audio2 = cp.array([])

    result = correlation_with_sync_moving_window(audio1, audio2)

    assert result.shape == (0, 2)


def test_returns_array_of_correct_shape():
    audio1 = cp.random.default_rng(0).random(PRECISION_BEATS * 10)
    audio2 = cp.random.default_rng(1).random(PRECISION_BEATS * 10)

    result = correlation_with_sync_moving_window(audio1, audio2)

    assert result.shape[0] == audio1.shape[0] // PRECISION_BEATS
    assert result.shape[1] == 2


# noinspection PyTypeChecker
def test_calculates_correlation():
    common_part_offset = 50
    common_part_duration = 10
    audio1 = cp.random.default_rng(0).random(PRECISION_BEATS * 100)
    audio2 = cp.random.default_rng(1).random(PRECISION_BEATS * 100)
    audio2[common_part_offset * PRECISION_BEATS:(common_part_offset + common_part_duration) * PRECISION_BEATS] = \
        audio1[common_part_offset * PRECISION_BEATS:(common_part_offset + common_part_duration) * PRECISION_BEATS]

    result = correlation_with_sync_moving_window(audio1, audio2).get()

    values = result[:, 1]
    mean_correlation = np.mean(values)

    peak_mask = np.zeros(result.shape[0], dtype=bool)
    peak_mask[common_part_offset:common_part_offset + common_part_duration] = True
    assert np.all(values[peak_mask] > mean_correlation), \
        "All peak correlations should be greater than mean correlation"

    non_peak_mask = ~peak_mask
    assert np.all(values[non_peak_mask] < mean_correlation), \
        "All non-peak correlations should be less than mean correlation"
