# -*- coding: utf-8 -*-
"""
series_intro_recognizer

Comparing episodes of a series to find the opening/endings of the series.

This library receives a list of episodes, extracts the audio features of each
episode and compares them to find the common part of the series.

To reduce the number of comparisons, the library compares 4 sequential episodes.
The number of episodes to be compared can be changed in configuration.

See README.md for more information.
"""

from series_intro_recognizer.tp.interval import Interval
from series_intro_recognizer.processors.audio_samples import recognise_from_audio_samples
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files
