# Description: Configuration file for the series opening recognizer

# Audio sample rate.
RATE = 44100

# Minimum length of the intro in seconds.
MIN_SEGMENT_LENGTH_SEC = 30
MIN_SEGMENT_LENGTH_BEATS = int(MIN_SEGMENT_LENGTH_SEC * RATE)

# Precision of the correlation in seconds.
PRECISION_SECS = 1
PRECISION_BEATS = int(PRECISION_SECS * RATE)

# Number of sequential audio samples to be matched with each other.
# E.g. 5 means that the first sample will be matched with the next 5 samples.
SERIES_WINDOW = 5

# Number of sequential 'non-intro' beats that signal the end of the intro.
# Intro is considered to be over if the number of sequential 'non-intro' beats is greater than this value.
OFFSET_SEARCHER__SEQUENTIAL_BEATS = 30

# Save the correlation results to 'correlations' folder.
# Make sure to create the folder before running the app.
DEBUG_SAVE_CORRELATIONS = False
