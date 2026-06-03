import math

from series_intro_recognizer.config import Config
from series_intro_recognizer.tp.interval import Interval


def _filter_by_length(interval: Interval, cfg: Config) -> Interval:
    """
    If the interval is too long or short, it is replaced with NaNs.
    """
    length = interval.end - interval.start
    return (interval
            if cfg.min_intro_length_secs <= length <= cfg.max_intro_length_secs
            else Interval(math.nan, math.nan))


def _adjust_borders(interval: Interval, audio_duration: float, cfg: Config) -> Interval:
    """
    If the interval is too close to the beginning or the end of the audio,
    it adjusts the interval to the beginning or the end of the audio.
    """
    if cfg.adjustment_threshold is False:
        return interval

    start = 0 \
        if interval.start - cfg.adjustment_threshold_secs <= 0 \
        else interval.start

    end = audio_duration \
        if interval.end + cfg.adjustment_threshold_secs >= audio_duration \
        else interval.end

    return Interval(start, end)


def improve_interval(interval: Interval, audio_duration: float, cfg: Config) -> Interval:
    if math.isnan(interval.start) or math.isnan(interval.end):
        return interval

    interval = _adjust_borders(interval, audio_duration, cfg)
    interval = _filter_by_length(interval, cfg)

    return interval
