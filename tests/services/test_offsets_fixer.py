import math

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.interval_improver import improve_interval
from series_intro_recognizer.tp.interval import Interval


def test__nan_interval__no_changes():
    cfg = Config()
    interval = Interval(math.nan, math.nan)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == interval


def test__regular_interval__no_changes():
    cfg = Config()
    interval = Interval(30, 50)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == interval


def test__interval_too_long__replaced_with_nans():
    cfg = Config()
    cfg.MAX_SEGMENT_LENGTH_SEC = 100
    interval = Interval(30, 150)
    audio_duration = 1000

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert math.isnan(fixed_interval.start)
    assert math.isnan(fixed_interval.end)


def test__interval_close_to_start__adjusted_to_start():
    cfg = Config()
    cfg.ADJUSTMENT_THRESHOLD_SECS = 10
    interval = Interval(10, 50)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == Interval(0, 50)


def test__interval_close_to_end__adjusted_to_end():
    cfg = Config()
    cfg.ADJUSTMENT_THRESHOLD_SECS = 10
    interval = Interval(30, 90)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == Interval(30, 100)


def test__interval_close_to_start_and_end__adjusted_to_start_and_end():
    cfg = Config()
    cfg.ADJUSTMENT_THRESHOLD_SECS = 10
    interval = Interval(10, 90)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == Interval(0, 100)
