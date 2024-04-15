import cupy as cp
import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.correlator.async_correlator import correlation_with_async_moving_window


def test_return_type():
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 3)
    audio2 = cp.random.default_rng(1).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 4)

    result = correlation_with_async_moving_window(audio1, audio2, cfg)

    assert isinstance(result, cp.ndarray)
    assert isinstance(result[0, 0], cp.ndarray)
    assert isinstance(result[0, 1], cp.ndarray)
    assert isinstance(result[0, 2], cp.ndarray)


def test_return_cupy_data():
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 3)
    audio2 = cp.random.default_rng(1).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 4)

    result = correlation_with_async_moving_window(audio1, audio2, cfg)

    assert cp == cp.get_array_module(result)
    assert cp == cp.get_array_module(result[0, 0])
    assert cp == cp.get_array_module(result[0, 1])
    assert cp == cp.get_array_module(result[0, 2])


def test_returns_array():
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 3)
    audio2 = cp.random.default_rng(1).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 4)

    result = correlation_with_async_moving_window(audio1, audio2, cfg)

    assert 3 == result.shape[1]


def test_throws_error_on_cpu_array():
    cfg = Config()
    audio1 = np.random.default_rng(0).random(10000)
    audio2 = np.random.default_rng(1).random(10000)

    try:
        correlation_with_async_moving_window(audio1, audio2, cfg)
        assert False
    except ValueError:
        pass


def test_throws_error_on_small_audio():
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(0)
    audio2 = cp.random.default_rng(1).random(0)

    try:
        correlation_with_async_moving_window(audio1, audio2, cfg)
        assert False
    except ValueError:
        pass


def test_splits_audio_correctly():
    cfg = Config()
    duration = 10
    audio1 = cp.random.default_rng(0).random(cfg.MIN_SEGMENT_LENGTH_BEATS * duration)
    audio2 = cp.random.default_rng(1).random(cfg.MIN_SEGMENT_LENGTH_BEATS * duration)

    result = correlation_with_async_moving_window(audio1, audio2, cfg)

    assert duration == result.shape[0]
    for i in range(duration):
        assert i * cfg.MIN_SEGMENT_LENGTH_BEATS == result[i, 0]


def test_calculates_correctly():
    """
    Test if the function calculates the correlation correctly.
    It generates two random audio arrays with the same segment in the 2/3 of the second audio.
    The correlation should be the highest for this segment.
    :return:
    """
    cfg = Config()
    audio1_expected_offset = 2 * cfg.MIN_SEGMENT_LENGTH_BEATS
    audio2_expected_offset = 7 * cfg.MIN_SEGMENT_LENGTH_BEATS

    audio1 = cp.random.default_rng(1).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 10)
    audio2 = cp.random.default_rng(2).random(cfg.MIN_SEGMENT_LENGTH_BEATS * 10)
    common_segment = cp.random.default_rng(3).random(cfg.MIN_SEGMENT_LENGTH_BEATS)
    audio1[audio1_expected_offset:audio1_expected_offset + cfg.MIN_SEGMENT_LENGTH_BEATS] = common_segment
    audio2[audio2_expected_offset:audio2_expected_offset + cfg.MIN_SEGMENT_LENGTH_BEATS] = common_segment

    result = correlation_with_async_moving_window(audio1, audio2, cfg).get()

    # Check that the offset of the 2nd audio is the expected one
    assert audio2_expected_offset == result[audio1_expected_offset // cfg.MIN_SEGMENT_LENGTH_BEATS, 1]

    # Check that every correlation is lower than the expected one
    for i in range(result.shape[0]):
        if i != audio1_expected_offset // cfg.MIN_SEGMENT_LENGTH_BEATS:
            assert result[i, 2] < result[audio1_expected_offset // cfg.MIN_SEGMENT_LENGTH_BEATS, 2]
