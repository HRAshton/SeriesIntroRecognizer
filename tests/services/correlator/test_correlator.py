import math

import cupy as cp
import numpy as np

from series_opening_recognizer.configuration import RATE
from series_opening_recognizer.services.correlator.correlator import calculate_correlation


def test_integration_returns_none_when_no_correlation():
    audio1 = cp.random.default_rng(0).random(1000)
    audio2 = cp.random.default_rng(0).random(1000)

    result = calculate_correlation(audio1, audio2)

    assert result is None


# noinspection PyTypeChecker
def test_integration_calculates_correctly():
    offset1 = int(4.2 * RATE)
    offset2 = int(7.3 * RATE)
    common_part_size = int(2.2 * RATE)

    audio1 = cp.random.default_rng(0).random(RATE * 30)
    audio2 = cp.random.default_rng(1).random(RATE * 45)
    common_part = cp.random.default_rng(2).random(common_part_size)
    audio1[offset1:offset1 + common_part.size] = common_part
    audio2[offset2:offset2 + common_part.size] = common_part

    result = calculate_correlation(audio1, audio2)

    assert cp.isclose(result[0], cp.array(0))
    assert cp.isclose(result[1], cp.array(3.1))
    assert result[2].shape[0] == 26  # What is the size of the result[2]?
    assert result[2].shape[1] == 2

    expected_beat_values = cp.arange(0, 26, 1) * RATE
    observed_beat_values = result[2][:, 0].get()
    assert np.allclose(observed_beat_values, expected_beat_values)

    values = result[2][:, 1].get()
    mean = np.mean(values)

    peak_mask = np.zeros(values.shape[0], dtype=bool)
    peak_mask[math.floor(offset1 / RATE):math.ceil((offset1 + common_part_size) / RATE)] = True
    assert np.all(values[peak_mask] > mean)

    non_peak_mask = ~peak_mask
    assert np.all(values[non_peak_mask] < mean)
