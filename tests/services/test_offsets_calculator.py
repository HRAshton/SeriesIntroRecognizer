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
    low1 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 3)
    high = cp.ones(cfg.offset_calculator_max_gap_intervals * 4) * 12
    low2 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 4)
    corr_values = cp.concatenate([low1, high, low2], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (3 * cfg.offset_calculator_max_gap_intervals,
                                   (3 + 4) * cfg.offset_calculator_max_gap_intervals)


def test__plateau_with_gaps__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 3)
    high1 = cp.ones(cfg.offset_calculator_max_gap_intervals * 4) * 12
    low2 = cp.ones(cfg.offset_calculator_max_gap_intervals)  # short gap
    high2 = cp.ones(cfg.offset_calculator_max_gap_intervals * 3) * 12
    low3 = cp.zeros(cfg.offset_calculator_max_gap_intervals + 1)  # long gap
    high3 = cp.ones(cfg.offset_calculator_max_gap_intervals * 4) * 12
    low4 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 10)
    corr_values = cp.concatenate([low1, high1, low2, high2, low3, high3, low4], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (3 * cfg.offset_calculator_max_gap_intervals,
                                   (3 + 4 + 1 + 3) * cfg.offset_calculator_max_gap_intervals)


def test__longest_sequence_without_min_continuous_positive_secs__returns_none() -> None:
    cfg = Config(
        offset_calculator_max_gap_secs=9,
        precision_secs=1,
        offset_calculator_min_continuous_positive_secs=2,
    )
    positive_spikes = []
    for _ in range(20):
        positive_spikes.append(cp.ones(1, dtype=cp.float32) * 12)
        positive_spikes.append(cp.zeros(cfg.offset_calculator_max_gap_intervals, dtype=cp.float32))
    corr_values = cp.concatenate(positive_spikes, dtype=cp.float32)

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result is None


def test__longest_sequence_uses_configured_continuous_positive_secs() -> None:
    cfg = Config(
        offset_calculator_max_gap_secs=9,
        precision_secs=1,
        offset_calculator_min_continuous_positive_secs=1,
    )
    positive_spikes = []
    for _ in range(20):
        positive_spikes.append(cp.ones(1, dtype=cp.float32) * 12)
        positive_spikes.append(cp.zeros(cfg.offset_calculator_max_gap_intervals, dtype=cp.float32))
    corr_values = cp.concatenate(positive_spikes, dtype=cp.float32)

    find_offsets_result = find_offsets(corr_values, cfg)

    expected_end = corr_values.size - cfg.offset_calculator_max_gap_intervals
    assert find_offsets_result == (0, expected_end)


def test__longest_sequence_skips_candidate_without_min_continuous_positive_secs() -> None:
    cfg = Config(
        offset_calculator_max_gap_secs=9,
        precision_secs=1,
        offset_calculator_min_continuous_positive_secs=2,
    )
    noisy_long_candidate = []
    for _ in range(20):
        noisy_long_candidate.append(cp.ones(1, dtype=cp.float32) * 12)
        noisy_long_candidate.append(cp.zeros(cfg.offset_calculator_max_gap_intervals, dtype=cp.float32))
    gap = cp.zeros(cfg.offset_calculator_max_gap_intervals + 1, dtype=cp.float32)
    valid_short_candidate = cp.ones(4, dtype=cp.float32) * 12
    low = cp.zeros(cfg.offset_calculator_max_gap_intervals * 2, dtype=cp.float32)
    corr_values = cp.concatenate([*noisy_long_candidate, gap, valid_short_candidate, low], dtype=cp.float32)

    find_offsets_result = find_offsets(corr_values, cfg)

    start = sum(part.size for part in noisy_long_candidate) + gap.size
    assert find_offsets_result == (start, start + valid_short_candidate.size)


def test__trailing_short_positive_run_after_gap_can_be_ignored() -> None:
    cfg = Config(
        offset_calculator_max_gap_secs=5,
        precision_secs=1,
        offset_calculator_min_continuous_positive_secs=2,
        offset_calculator_max_ignored_trailing_positive_secs=3,
    )
    ending = cp.ones(30, dtype=cp.float32) * 12
    scene_after_ending = cp.zeros(3, dtype=cp.float32)
    endcard = cp.ones(2, dtype=cp.float32) * 12
    low = cp.zeros(cfg.offset_calculator_max_gap_intervals + 1, dtype=cp.float32)
    corr_values = cp.concatenate([ending, scene_after_ending, endcard, low], dtype=cp.float32)

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (0, ending.size)


def test__trailing_short_positive_run_is_kept_when_ignoring_is_disabled() -> None:
    cfg = Config(
        offset_calculator_max_gap_secs=5,
        precision_secs=1,
        offset_calculator_min_continuous_positive_secs=2,
    )
    ending = cp.ones(30, dtype=cp.float32) * 12
    scene_after_ending = cp.zeros(3, dtype=cp.float32)
    endcard = cp.ones(2, dtype=cp.float32) * 12
    low = cp.zeros(cfg.offset_calculator_max_gap_intervals + 1, dtype=cp.float32)
    corr_values = cp.concatenate([ending, scene_after_ending, endcard, low], dtype=cp.float32)

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (0, ending.size + scene_after_ending.size + endcard.size)


def test__trailing_positive_run_longer_than_endcard_limit_is_kept() -> None:
    cfg = Config(
        offset_calculator_max_gap_secs=5,
        precision_secs=1,
        offset_calculator_min_continuous_positive_secs=2,
        offset_calculator_max_ignored_trailing_positive_secs=3,
    )
    ending = cp.ones(30, dtype=cp.float32) * 12
    scene_after_ending = cp.zeros(3, dtype=cp.float32)
    repeated_scene = cp.ones(4, dtype=cp.float32) * 12
    low = cp.zeros(cfg.offset_calculator_max_gap_intervals + 1, dtype=cp.float32)
    corr_values = cp.concatenate([ending, scene_after_ending, repeated_scene, low], dtype=cp.float32)

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (0, ending.size + scene_after_ending.size + repeated_scene.size)


def test__plateau_with_extreme_high_peaks__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 3)
    high = cp.ones(cfg.offset_calculator_max_gap_intervals * 4) * 12
    low2 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 4)
    corr_values = cp.concatenate([low1, high, low2], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    # Add extreme high peaks
    start = int(3.1 * cfg.offset_calculator_max_gap_intervals)
    end = int(3.7 * cfg.offset_calculator_max_gap_intervals)
    corr_values[start:end] = 10000

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (3 * cfg.offset_calculator_max_gap_intervals,
                                   (3 + 4) * cfg.offset_calculator_max_gap_intervals)


def test__empty_array__returns_none() -> None:
    cfg = Config()
    corr_values = cp.array([], dtype=cp.float32)

    result = find_offsets(corr_values, cfg)

    assert result is None


def test__plateau_at_start__correct_offsets() -> None:
    """High values at the very beginning of the array (start index = 0)."""
    cfg = Config()
    n = cfg.offset_calculator_max_gap_intervals
    high = cp.ones(n * 4, dtype=cp.float32) * 12
    low = cp.zeros(n * 8, dtype=cp.float32)
    corr_values = cp.concatenate([high, low])
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    result = find_offsets(corr_values, cfg)

    assert result == (0, 4 * n)


def test__two_plateaus_second_longer__returns_second() -> None:
    """Two plateaus separated by a gap larger than max_gap; the longer second one wins."""
    cfg = Config()
    n = cfg.offset_calculator_max_gap_intervals
    # layout:  [low1 | high1 | gap>n | high2 | low3]
    #  indices: [0,2n) [2n,4n) [4n,6n) [6n,10n) [10n,14n)
    low1  = cp.zeros(n * 2, dtype=cp.float32)
    high1 = cp.ones(n * 2, dtype=cp.float32) * 12   # length 2n
    low2  = cp.zeros(n * 2, dtype=cp.float32)        # gap (2n > n) → splits sequences
    high2 = cp.ones(n * 4, dtype=cp.float32) * 12   # length 4n — strictly longer
    low3  = cp.zeros(n * 4, dtype=cp.float32)
    corr_values = cp.concatenate([low1, high1, low2, high2, low3])
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    result = find_offsets(corr_values, cfg)

    assert result == (6 * n, 10 * n)


def test__plateau_on_edge__correct_offsets() -> None:
    cfg = Config()
    low1 = cp.zeros(cfg.offset_calculator_max_gap_intervals * 8)
    high = cp.ones(cfg.offset_calculator_max_gap_intervals * 3) * 12
    corr_values = cp.concatenate([low1, high], dtype=cp.float32)
    corr_values += cp.random.rand(corr_values.shape[0]) * 0.1

    find_offsets_result = find_offsets(corr_values, cfg)

    assert find_offsets_result == (8 * cfg.offset_calculator_max_gap_intervals,
                                   (8 + 3) * cfg.offset_calculator_max_gap_intervals)
