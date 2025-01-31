import cupy as cp  # type: ignore
import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.correlator.sync_correlator import correlation_with_sync_moving_window


def test_return_type() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 10)
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 10)

    result = correlation_with_sync_moving_window(audio1, audio2, cfg)

    assert isinstance(result, cp.ndarray)
    assert isinstance(result[0, 0], cp.ndarray)
    assert isinstance(result[0, 1], cp.ndarray)


def test_return_cupy_data() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 10)
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 10)

    result = correlation_with_sync_moving_window(audio1, audio2, cfg)

    assert cp.get_array_module(result) == cp
    assert cp.get_array_module(result[0, 0]) == cp
    assert cp.get_array_module(result[0, 1]) == cp


def test_returns_array() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 10)
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 10)

    result = correlation_with_sync_moving_window(audio1, audio2, cfg)

    assert result.shape[1] == 2


def test_throws_error_on_cpu_array() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 10).get()
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 10).get()

    try:
        correlation_with_sync_moving_window(audio1, audio2, cfg)
        assert False
    except ValueError:
        assert True


def test_throws_error_on_audio1_longer_than_audio2() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 20)
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 10)

    try:
        correlation_with_sync_moving_window(audio1, audio2, cfg)
        assert False
    except ValueError:
        assert True


def test_returns_empty_array_on_empty_input() -> None:
    cfg = Config()
    audio1 = cp.array([])
    audio2 = cp.array([])

    result = correlation_with_sync_moving_window(audio1, audio2, cfg)

    assert result.shape == (0, 2)


def test_returns_array_of_correct_shape() -> None:
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 10)
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 10)

    result = correlation_with_sync_moving_window(audio1, audio2, cfg)

    assert result.shape[0] == audio1.shape[0] // cfg.precision_beats
    assert result.shape[1] == 2


# noinspection PyTypeChecker
def test_calculates_correlation() -> None:
    cfg = Config()
    common_part_offset = 50
    common_part_duration = 10
    audio1 = cp.random.default_rng(0).random(cfg.precision_beats * 100)
    audio2 = cp.random.default_rng(1).random(cfg.precision_beats * 100)
    common_part_start = common_part_offset * cfg.precision_beats
    common_part_end = (common_part_offset + common_part_duration) * cfg.precision_beats
    audio2[common_part_start:common_part_end] = audio1[common_part_start:common_part_end]

    result = correlation_with_sync_moving_window(audio1, audio2, cfg).get()

    values = result[:, 1]
    mean_correlation = np.mean(values)

    peak_mask = np.zeros(result.shape[0], dtype=bool)
    peak_mask[common_part_offset:common_part_offset + common_part_duration] = True
    assert np.all(values[peak_mask] > mean_correlation), \
        'All peak correlations should be greater than mean correlation'

    non_peak_mask = ~peak_mask
    assert np.all(values[non_peak_mask] < mean_correlation), \
        'All non-peak correlations should be less than mean correlation'
