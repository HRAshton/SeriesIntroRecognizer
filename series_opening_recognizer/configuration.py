# Description: Configuration file for the series opening recognizer

# Audio sample rate.
RATE = 44100

# Minimum length of the intro in seconds.
MIN_SEGMENT_LENGTH_SEC = 30
MIN_SEGMENT_LENGTH_BEATS = int(MIN_SEGMENT_LENGTH_SEC * RATE)

# Precision of the correlation in seconds.
PRECISION_SECS = .5
PRECISION_BEATS = int(PRECISION_SECS * RATE)

# Number of sequential audio samples to be matched with each other.
# E.g. 5 means that the first sample will be matched with the next 5 samples.
SERIES_WINDOW = 5

# Number of sequential 'non-intro' seconds that signal the end of the intro.
# Intro is considered to be over if the number of sequential 'non-intro' beats is greater than this value.
OFFSET_SEARCHER__SEQUENTIAL_SECS = 30
OFFSET_SEARCHER__SEQUENTIAL_INTERVALS = int(OFFSET_SEARCHER__SEQUENTIAL_SECS / PRECISION_SECS)

# Save the correlation results to 'correlations' folder.
# Make sure to create the folder before running the app.
DEBUG_SAVE_INTERMEDIATE_RESULTS = False


def set_rate(rate: int) -> None:
    global RATE
    RATE = rate


def set_min_segment_length_sec(min_segment_length_sec: int) -> None:
    global MIN_SEGMENT_LENGTH_SEC
    global MIN_SEGMENT_LENGTH_BEATS
    MIN_SEGMENT_LENGTH_SEC = min_segment_length_sec
    MIN_SEGMENT_LENGTH_BEATS = int(MIN_SEGMENT_LENGTH_SEC * RATE)


def set_precision_secs(precision_secs: int) -> None:
    global PRECISION_SECS
    global PRECISION_BEATS
    PRECISION_SECS = precision_secs
    PRECISION_BEATS = int(PRECISION_SECS * RATE)


def set_series_window(series_window: int) -> None:
    global SERIES_WINDOW
    SERIES_WINDOW = series_window


def set_offset_searcher_sequential_beats(offset_searcher_sequential_beats: int) -> None:
    global OFFSET_SEARCHER__SEQUENTIAL_SECS
    OFFSET_SEARCHER__SEQUENTIAL_SECS = offset_searcher_sequential_beats


def set_debug_save_correlations(debug_save_correlations: bool) -> None:
    global DEBUG_SAVE_INTERMEDIATE_RESULTS
    DEBUG_SAVE_INTERMEDIATE_RESULTS = debug_save_correlations
