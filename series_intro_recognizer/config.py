# pylint: disable=too-many-instance-attributes
"""
Configuration module for the Series Intro Recognizer.

This module defines a `Config` class that stores and manages
various parameters used for audio processing. It includes
default values, computed attributes, and documentation
for better maintainability.

Usage:
    from series_intro_recognizer.config import Config
    config = Config()
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configuration class for the series opening recognizer.

    This class stores various parameters used for audio processing, such as
    sample rate, segment lengths, precision, and threshold values. It also
    computes dependent attributes like segment lengths in beats and
    offset intervals.

    Attributes:
        rate (int): Audio sample rate (Hz).
        min_segment_length_sec (int): Minimum length of the intro in seconds.
        max_segment_length_sec (int): Maximum length of the intro in seconds.
        precision_secs (float): Precision of the correlation in seconds.
        series_window (int): Number of sequential audio samples to be matched.
        offset_searcher_sequential_secs (int): Number of sequential 'non-intro'
            seconds that signal the end of the intro.
        adjustment_threshold (bool): Whether to adjust the intro borders.
        adjustment_threshold_secs (float): Threshold for border adjustment.
        save_intermediate_results (bool): Whether to save correlation results.

    Computed Properties:
        min_segment_length_beats (int): Minimum length of the intro in beats.
        max_segment_length_beats (int): Maximum length of the intro in beats.
        precision_beats (int): Precision of the correlation in beats.
        offset_searcher_sequential_intervals (int): Number of sequential
            'non-intro' beats that signal the end of the intro.
    """

    rate: int = 44100  # Audio sample rate

    min_segment_length_sec: int = 30  # Minimum length of the intro (seconds)
    max_segment_length_sec: int = 150  # Maximum length of the intro (seconds)
    precision_secs: float = 0.5  # Precision of the correlation (seconds)

    series_window: int = 5  # Number of sequential audio samples to be matched

    offset_searcher_sequential_secs: int = 30  # 'Non-intro' seconds that signal the end of the intro

    adjustment_threshold: bool = True  # Whether to adjust intro borders
    adjustment_threshold_secs: float = 3.0  # Threshold for border adjustment

    save_intermediate_results: bool = False  # Save correlation results

    # Computed attributes
    _min_segment_length_beats: int = field(init=False)
    _max_segment_length_beats: int = field(init=False)
    _precision_beats: int = field(init=False)
    _offset_searcher_sequential_intervals: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Compute dependent attributes after initialization.
        """
        self._min_segment_length_beats = int(self.min_segment_length_sec * self.rate)
        self._max_segment_length_beats = int(self.max_segment_length_sec * self.rate)
        self._precision_beats = int(self.precision_secs * self.rate)
        self._offset_searcher_sequential_intervals = int(self.offset_searcher_sequential_secs / self.precision_secs)

    @property
    def min_segment_length_beats(self) -> int:
        """Returns the minimum length of the intro in beats."""
        return self._min_segment_length_beats

    @property
    def max_segment_length_beats(self) -> int:
        """Returns the maximum length of the intro in beats."""
        return self._max_segment_length_beats

    @property
    def precision_beats(self) -> int:
        """Returns the precision of the correlation in beats."""
        return self._precision_beats

    @property
    def offset_searcher_sequential_intervals(self) -> int:
        """Returns the number of sequential 'non-intro' beats that signal the end of the intro."""
        return self._offset_searcher_sequential_intervals
