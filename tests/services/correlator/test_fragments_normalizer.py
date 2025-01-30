import cupy as cp  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.correlator.fragments_normalizer import align_fragments


def test_return_type():
    cfg = Config()
    best_offset1 = cp.array(0 * cfg.RATE)
    best_offset2 = cp.array(2 * cfg.RATE)
    audio1 = cp.random.default_rng(0).random(10 * cfg.RATE)
    audio2 = cp.random.default_rng(0).random(10 * cfg.RATE)

    truncated_audio1, truncated_audio2, offset1_secs, offset2_secs = \
        align_fragments(best_offset1, best_offset2, audio1, audio2, cfg)

    assert isinstance(truncated_audio1, cp.ndarray)
    assert isinstance(truncated_audio2, cp.ndarray)
    assert isinstance(offset1_secs, cp.ndarray)
    assert isinstance(offset2_secs, cp.ndarray)


def test_return_cupy_data():
    cfg = Config()
    best_offset1 = cp.array(1 * cfg.RATE)
    best_offset2 = cp.array(0 * cfg.RATE)
    audio1 = cp.random.default_rng(0).random(10 * cfg.RATE)
    audio2 = cp.random.default_rng(0).random(10 * cfg.RATE)

    truncated_audio1, truncated_audio2, offset1_secs, offset2_secs = \
        align_fragments(best_offset1, best_offset2, audio1, audio2, cfg)

    assert cp.get_array_module(truncated_audio1) == cp
    assert cp.get_array_module(truncated_audio2) == cp
    assert cp.get_array_module(offset1_secs) == cp
    assert cp.get_array_module(offset2_secs) == cp


def test_return_array_of_expected_shape():
    cfg = Config()
    best_offset1 = cp.array(17 * cfg.RATE)
    best_offset2 = cp.array(11 * cfg.RATE)
    audio1 = cp.random.default_rng(0).random(30 * cfg.RATE)
    audio2 = cp.random.default_rng(0).random(31 * cfg.RATE)

    truncated_audio1, truncated_audio2, offset1_secs, offset2_secs = \
        align_fragments(best_offset1, best_offset2, audio1, audio2, cfg)

    expected_length = (30 - (17 - 11)) * cfg.RATE
    assert truncated_audio1.shape == (expected_length,)
    assert truncated_audio2.shape == (expected_length,)
    assert offset1_secs.shape == ()
    assert offset2_secs.shape == ()


def test_align_fragments():
    cfg = Config()
    offset1_of_common_fragment = 3
    offset2_of_common_fragment = 1
    best_offset1 = cp.array(offset1_of_common_fragment * cfg.RATE)
    best_offset2 = cp.array(offset2_of_common_fragment * cfg.RATE)
    audio1 = cp.arange(0, 10 * cfg.RATE, 1.0)
    audio2 = cp.arange(0, 12 * cfg.RATE, 1.0)

    truncated_audio1, truncated_audio2, offset1_secs, offset2_secs = \
        align_fragments(best_offset1, best_offset2, audio1, audio2, cfg)

    expected_length = (10 - (3 - 1)) * cfg.RATE
    assert offset1_secs == 2
    assert offset2_secs == 0
    assert cp.allclose(truncated_audio1, audio1[2 * cfg.RATE:2 * cfg.RATE + expected_length])
    assert cp.allclose(truncated_audio2, audio2[0:expected_length])
