import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_samples import recognise_from_audio_samples


def test__no_audios__returns_empty() -> None:
    cfg = Config()

    result = recognise_from_audio_samples([], cfg)

    assert result == []


def test__one_audio__returns_empty() -> None:
    cfg = Config()

    result = recognise_from_audio_samples([np.zeros(1)], cfg)

    assert result == []


def test__empty_audio__throws_error() -> None:
    cfg = Config()

    try:
        recognise_from_audio_samples([np.zeros(0)], cfg)
        assert False
    except ValueError as e:
        assert str(e) == 'Empty audio passed.'


def test__two_short_audios__returns_nan() -> None:
    cfg = Config()

    result = recognise_from_audio_samples([np.zeros(1), np.zeros(1)], cfg)

    for res in result:
        assert np.isnan(res.start)
        assert np.isnan(res.end)


def test__same_audios__returns_nan() -> None:
    cfg = Config()
    audio = np.random.default_rng(0).random(cfg.MIN_SEGMENT_LENGTH_BEATS)

    result = recognise_from_audio_samples([audio, audio], cfg)

    for res in result:
        assert np.isnan(res.start)
        assert np.isnan(res.end)


def test__random_audios__returns_nan() -> None:
    cfg = Config()
    audio1 = np.random.default_rng(0).random(cfg.MIN_SEGMENT_LENGTH_BEATS)
    audio2 = np.random.default_rng(1).random(cfg.MIN_SEGMENT_LENGTH_BEATS)

    result = recognise_from_audio_samples([audio1, audio2], cfg)

    for res in result:
        assert np.isnan(res.start)
        assert np.isnan(res.end)
