import math

from series_intro_recognizer.config import Config
from series_intro_recognizer.tp.interval import Interval


def _filter_too_long_intervals(interval: Interval, cfg: Config) -> Interval:
    """
    If the interval is too long, it is replaced with NaNs.
    """
    return (interval
            if interval.end - interval.start <= cfg.MAX_SEGMENT_LENGTH_SEC
            else Interval(math.nan, math.nan))


def _adjust_borders(interval: Interval, audio_duration: float, cfg: Config) -> Interval:
    """
    If the interval is too close to the beginning or the end of the audio,
    it adjusts the interval to the beginning or the end of the audio.
    """
    start = 0 \
        if interval.start - cfg.ADJUSTMENT_THRESHOLD_SECS <= 0 \
        else interval.start

    end = audio_duration \
        if interval.end + cfg.ADJUSTMENT_THRESHOLD_SECS >= audio_duration \
        else interval.end

    return Interval(start, end)


def improve_interval(interval: Interval, audio_duration: float, cfg: Config) -> Interval:
    if math.isnan(interval.start) or math.isnan(interval.end):
        return interval

    interval = _adjust_borders(interval, audio_duration, cfg)
    interval = _filter_too_long_intervals(interval, cfg)

    return interval
