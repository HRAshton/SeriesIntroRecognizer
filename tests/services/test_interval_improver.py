import math

from series_intro_recognizer.config import Config
from series_intro_recognizer.services.interval_improver import improve_interval
from series_intro_recognizer.tp.interval import Interval


def test__nan_interval__no_changes() -> None:
    cfg = Config()
    interval = Interval(math.nan, math.nan)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == interval


def test__regular_interval__no_changes() -> None:
    cfg = Config()
    interval = Interval(30, 50)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == interval


def test__interval_too_long__replaced_with_nans() -> None:
    cfg = Config()
    cfg.max_segment_length_sec = 100
    interval = Interval(30, 150)
    audio_duration = 1000

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert math.isnan(fixed_interval.start)
    assert math.isnan(fixed_interval.end)


def test__interval_close_to_start_but_disabled__no_adjustment() -> None:
    cfg = Config()
    cfg.adjustment_threshold = False
    cfg.adjustment_threshold_secs = 10
    interval = Interval(10, 50)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == interval


def test__interval_close_to_start__adjusted_to_start() -> None:
    cfg = Config()
    cfg.adjustment_threshold_secs = 10
    interval = Interval(10, 50)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == Interval(0, 50)


def test__interval_close_to_end__adjusted_to_end() -> None:
    cfg = Config()
    cfg.adjustment_threshold_secs = 10
    interval = Interval(30, 90)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == Interval(30, 100)


def test__interval_close_to_start_and_end__adjusted_to_start_and_end() -> None:
    cfg = Config()
    cfg.adjustment_threshold_secs = 10
    interval = Interval(10, 90)
    audio_duration = 100

    fixed_interval = improve_interval(interval, audio_duration, cfg)

    assert fixed_interval == Interval(0, 100)
