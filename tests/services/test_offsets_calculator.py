import cupy as cp  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.offsets_calculator import find_offsets


def test__same_values__not_enough_correlation() -> None:
    cfg = Config()
    corr_values = cp.array([10000, 10000, 10000, 10000, 10000])

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result is None


def test__plateau__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_searcher_sequential_intervals * 3)
    high = cp.ones(cfg.offset_searcher_sequential_intervals * 4) * 12
    low2 = cp.zeros(cfg.offset_searcher_sequential_intervals * 4)
    corr_values = cp.concatenate([low1, high, low2], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (3 * cfg.offset_searcher_sequential_intervals,
                                   (3 + 4) * cfg.offset_searcher_sequential_intervals)


def test__plateau_with_gaps__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_searcher_sequential_intervals * 3)
    high1 = cp.ones(cfg.offset_searcher_sequential_intervals * 4) * 12
    low2 = cp.ones(cfg.offset_searcher_sequential_intervals)  # short gap
    high2 = cp.ones(cfg.offset_searcher_sequential_intervals * 3) * 12
    low3 = cp.zeros(cfg.offset_searcher_sequential_intervals + 1)  # long gap
    high3 = cp.ones(cfg.offset_searcher_sequential_intervals * 4) * 12
    low4 = cp.zeros(cfg.offset_searcher_sequential_intervals * 10)
    corr_values = cp.concatenate([low1, high1, low2, high2, low3, high3, low4], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (3 * cfg.offset_searcher_sequential_intervals,
                                   (3 + 4 + 1 + 3) * cfg.offset_searcher_sequential_intervals)


def test__plateau_with_extreme_high_peaks__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_searcher_sequential_intervals * 3)
    high = cp.ones(cfg.offset_searcher_sequential_intervals * 4) * 12
    low2 = cp.zeros(cfg.offset_searcher_sequential_intervals * 4)
    corr_values = cp.concatenate([low1, high, low2], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    # Add extreme high peaks
    start = int(3.1 * cfg.offset_searcher_sequential_intervals)
    end = int(3.7 * cfg.offset_searcher_sequential_intervals)
    corr_values[start:end] = 10000

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (3 * cfg.offset_searcher_sequential_intervals,
                                   (3 + 4) * cfg.offset_searcher_sequential_intervals)


def test__plateau_on_edge__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_searcher_sequential_intervals * 8)
    high = cp.ones(cfg.offset_searcher_sequential_intervals * 3) * 12
    corr_values = cp.concatenate([low1, high], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (8 * cfg.offset_searcher_sequential_intervals,
                                   (8 + 3) * cfg.offset_searcher_sequential_intervals)
