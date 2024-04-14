import cupy as cp

from series_opening_recognizer.configuration import OFFSET_SEARCHER__SEQUENTIAL_INTERVALS
from series_opening_recognizer.services.offsets_calculator import find_offsets


def test__same_values__not_enough_correlation():
    corr_values = cp.array([10000, 10000, 10000, 10000, 10000])

    find_offsets_result = find_offsets(corr_values)

    assert find_offsets_result is None


def test__random_values__not_enough_correlation():
    corr_values = cp.random.rand(10)

    find_offsets_result = find_offsets(corr_values)

    assert find_offsets_result is None


def test__sequential_values__not_enough_correlation():
    corr_values = cp.array([i for i in range(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 2)])

    find_offsets_result = find_offsets(corr_values)

    assert find_offsets_result is None


def test__plateau__correct_offsets():
    low1 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 3)
    high = cp.ones(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 4) * 12
    low2 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 4)
    corr_values = cp.concatenate([low1, high, low2], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values)

    assert find_offsets_result == (3 * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS,
                                   (3 + 4) * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS)


def test__plateau_with_gaps__correct_offsets():
    low1 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 3)
    high1 = cp.ones(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 4) * 12
    low2 = cp.ones(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS - 1)  # short gap
    high2 = cp.ones(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 3) * 12
    low3 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS)  # long gap
    high3 = cp.ones(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 4) * 12
    low4 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 10)
    corr_values = cp.concatenate([low1, high1, low2, high2, low3, high3, low4], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values)

    assert find_offsets_result == (3 * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS,
                                   (3 + 4 + 1 + 3) * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS - 1)


def test__plateau_with_extreme_high_peaks__correct_offsets():
    low1 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 3)
    high = cp.ones(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 4) * 12
    low2 = cp.zeros(OFFSET_SEARCHER__SEQUENTIAL_INTERVALS * 4)
    corr_values = cp.concatenate([low1, high, low2], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    # Add extreme high peaks
    start = 3.1 * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS
    end = 3.7 * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS
    corr_values[start:end] = 10000

    find_offsets_result = find_offsets(corr_values)

    assert find_offsets_result == (3 * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS,
                                   (3 + 4) * OFFSET_SEARCHER__SEQUENTIAL_INTERVALS)
