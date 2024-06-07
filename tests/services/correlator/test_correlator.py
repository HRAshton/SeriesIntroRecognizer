import math

import cupy as cp
import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.correlator.correlator import calculate_correlation


def test_integration_returns_none_when_no_correlation():
    cfg = Config()
    audio1 = cp.random.default_rng(0).random(1000)
    audio2 = cp.random.default_rng(0).random(1000)

    result = calculate_correlation(audio1, audio2, cfg)

    assert result is None


def test_integration_calculates_correctly():
    cfg = Config()
    offset1 = int(4.2 * cfg.RATE)
    offset2 = int(7.3 * cfg.RATE)
    common_part_size = int(2.2 * cfg.RATE)

    audio1 = cp.random.default_rng(0).random(cfg.RATE * 30)
    audio2 = cp.random.default_rng(1).random(cfg.RATE * 45)
    common_part = cp.random.default_rng(2).random(common_part_size)
    audio1[offset1:offset1 + common_part.size] = common_part
    audio2[offset2:offset2 + common_part.size] = common_part

    result = calculate_correlation(audio1, audio2, cfg)

    precision_beats_multiplier = cfg.RATE * cfg.PRECISION_SECS

    assert cp.isclose(result[0], cp.array(0)), 'Audio 1 should have a correct offset'
    assert cp.isclose(result[1], cp.array(3.1)), 'Audio 2 should have a correct offset'
    assert result[2].shape[0] == int((30 - (7.3 - 4.2)) / cfg.PRECISION_SECS), \
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
