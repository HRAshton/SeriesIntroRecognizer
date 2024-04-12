# SeriesOpeningRecogniser

Comparing episodes of a series to find the opening/endings of the series.

**WARN**: Do not pass the whole episodes, it will take a long time to process
and the results will not be accurate.

This library receives a list of episodes, extracts the audio features of each
episode and compares them to find the common part of the series.

To reduce the number of comparisons, the library compares 4 sequential episodes.
The number of episodes to be compared can be changed in configuration.

Input options:

- List of audio samples (numpy ndarrays)
- List of audio files

Output:

- List of intervals of the same fragment in the episodes

To find an opening, pass the first minutes (e.g. 5 minutes) of the episodes.
To find an ending, pass the last minutes (e.g. 5 minutes) of the episodes.

WARN: See the warning in the header.

## Usage

TODO