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

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Configuration class for the series opening recognizer.

    This class stores various parameters used for audio processing, such as
    sample rate, segment lengths, precision, and threshold values. It also
    exposes computed properties like segment lengths in beats and
    offset intervals.

    Attributes:
        rate (int): Audio sample rate (Hz).
        min_segment_length_sec (int): Minimum length of the intro in seconds.
        max_segment_length_sec (int): Maximum length of the intro in seconds.
        precision_secs (float): Precision of the correlation in seconds.
        series_window (int): Number of sequential audio samples to be matched.
        offset_searcher_sequential_secs (int): Number of sequential 'non-intro'
            seconds that signal the end of the intro.
        offset_searcher_similarity_too_close_coeff (float): Coefficient for determining
            if correlations are too close and should be skipped.
        adjustment_threshold (bool): Whether to adjust the intro borders.
        adjustment_threshold_secs (float): Threshold for border adjustment.
        save_intermediate_results (bool): Whether to save correlation results.
        correlator_always_choose_best_score (bool): Whether to always choose the best score instead of length clusters.

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
    offset_searcher_similarity_too_close_coeff: float = 1e-3  # Coefficient for determining if audios are the same

    correlator_always_choose_best_score: bool = False  # Whether to always choose the best score instead of length clusters

    adjustment_threshold: bool = True  # Whether to adjust intro borders
    adjustment_threshold_secs: float = 3.0  # Threshold for border adjustment

    save_intermediate_results: bool = False  # Save correlation results

    @property
    def min_segment_length_beats(self) -> int:
        """Returns the minimum length of the intro in beats."""
        return int(self.min_segment_length_sec * self.rate)

    @property
    def max_segment_length_beats(self) -> int:
        """Returns the maximum length of the intro in beats."""
        return int(self.max_segment_length_sec * self.rate)

    @property
    def precision_beats(self) -> int:
        """Returns the precision of the correlation in beats."""
        return int(self.precision_secs * self.rate)

    @property
    def offset_searcher_sequential_intervals(self) -> int:
        """Returns the number of sequential 'non-intro' beats that signal the end of the intro."""
        return int(self.offset_searcher_sequential_secs / self.precision_secs)

    @classmethod
    def preset_anime_opening(cls) -> "Config":
        """Preset for detecting anime openings (default values)."""
        return cls()

    @classmethod
    def preset_anime_ending(cls) -> "Config":
        """
        Preset for detecting anime endings (shorter non-intro window).
        Often anime episodes ends with the sequence "real ending"-"some scene"-"publisher's ad".
        Shorter threshold allows to detect this "scene" part.
        """
        return cls(offset_searcher_sequential_secs=5)

