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
        async_correlator_segment_secs (float): Fragment size for the async correlator (seconds).
            Also used as the minimum audio length required for processing.
        min_intro_length_secs (float): Minimum valid intro duration (seconds).
        max_intro_length_secs (float): Maximum valid intro duration (seconds).
        precision_secs (float): Precision of the correlation in seconds.
        series_window (int): Number of sequential audio samples to be matched.
        offset_calculator_max_gap_secs (int): Maximum gap in seconds allowed
            between above-threshold correlation values in a candidate sequence.
        offset_searcher_similarity_too_close_coeff (float): Coefficient for determining
            if correlations are too close and should be skipped.
        offset_calculator_min_continuous_positive_secs (float): Minimum continuous
            seconds in the found sequence that must remain above threshold.
        offset_calculator_max_ignored_trailing_positive_secs (float): Maximum trailing
            above-threshold seconds to trim from a candidate after an allowed gap.
        adjustment_threshold (bool): Whether to adjust the intro borders.
        adjustment_threshold_secs (float): Threshold for border adjustment.
        save_intermediate_results (bool): Whether to save correlation results.
        correlator_always_choose_best_score (bool): Whether to always choose the best score instead of length clusters.

    Computed Properties:
        min_segment_length_beats (int): Fragment size for the async correlator in samples.
        precision_beats (int): Precision of the correlation in beats.
        offset_calculator_max_gap_intervals (int): Maximum number of correlation
            intervals allowed between above-threshold values in a candidate sequence.
        offset_calculator_min_continuous_positive_intervals (int): Number of sequential
            above-threshold beats required inside the found sequence.
        offset_calculator_max_ignored_trailing_positive_intervals (int): Maximum trailing
            above-threshold intervals to trim from a candidate after an allowed gap.
    """

    rate: int = 44100  # Audio sample rate

    async_correlator_segment_secs: float = 30  # Fragment size for the async correlator (seconds)
    min_intro_length_secs: float = 30  # Minimum valid intro duration (seconds)
    max_intro_length_secs: float = 150  # Maximum valid intro duration (seconds)
    precision_secs: float = 0.5  # Precision of the correlation (seconds)

    series_window: int = 5  # Number of sequential audio samples to be matched

    offset_calculator_max_gap_secs: int = 30  # Maximum gap between positive spikes in a candidate sequence
    offset_searcher_similarity_too_close_coeff: float = 1e-3  # Coefficient for determining if audios are the same
    offset_calculator_min_continuous_positive_secs: float = 10  # Minimum continuous positive seconds in found sequence
    offset_calculator_max_ignored_trailing_positive_secs: float = 0  # Disabled unless a preset enables it

    correlator_always_choose_best_score: bool = False  # Whether to always choose the best score instead of length clusters

    adjustment_threshold: bool = True  # Whether to adjust intro borders
    adjustment_threshold_secs: float = 3.0  # Threshold for border adjustment

    save_intermediate_results: bool = False  # Save correlation results

    def __post_init__(self) -> None:
        if self.async_correlator_segment_secs > self.min_intro_length_secs:
            raise ValueError(
                f'async_correlator_segment_secs ({self.async_correlator_segment_secs}) '
                f'must be <= min_intro_length_secs ({self.min_intro_length_secs})'
            )

    @property
    def min_segment_length_beats(self) -> int:
        """Returns the async correlator fragment size in samples."""
        return int(self.async_correlator_segment_secs * self.rate)

    @property
    def precision_beats(self) -> int:
        """Returns the precision of the correlation in beats."""
        return int(self.precision_secs * self.rate)

    @property
    def offset_calculator_max_gap_intervals(self) -> int:
        """Returns the maximum gap between above-threshold correlation intervals."""
        return int(self.offset_calculator_max_gap_secs / self.precision_secs)

    @property
    def offset_calculator_min_continuous_positive_intervals(self) -> int:
        """Returns the number of sequential above-threshold beats required inside the found sequence."""
        return int(self.offset_calculator_min_continuous_positive_secs / self.precision_secs)

    @property
    def offset_calculator_max_ignored_trailing_positive_intervals(self) -> int:
        """Returns the maximum trailing above-threshold intervals to trim after an allowed gap."""
        return int(self.offset_calculator_max_ignored_trailing_positive_secs / self.precision_secs)

    @classmethod
    def preset_anime_opening(cls) -> "Config":
        """Preset for detecting anime openings (default values)."""
        return cls()

    @classmethod
    def preset_anime_ending(cls) -> "Config":
        """
        Preset for detecting anime endings (shorter non-intro window).
        Often anime episodes ends with the sequence "real ending"-"some scene"-"endcard".
        """
        return cls(
            offset_calculator_max_ignored_trailing_positive_secs=15,
        )
