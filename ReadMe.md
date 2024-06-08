# SeriesIntroRecognizer

Comparing episodes of a series to find the opening/endings of the series.

This library receives a list of episodes, extracts the audio features of each
episode and compares them to find the common part of the series.

To reduce the number of comparisons, the library compares 4 sequential episodes.
The number of episodes to be compared can be changed in configuration.

## Installation

The project uses Cupy. It supports both Cuda and AMD GPUs. Please refer to
the [Cupy installation guide](https://docs.cupy.dev/en/stable/install.html)
to install the correct version for your system.

```bash
pip install series_intro_recognizer
```

## Options

### Input options:

- Iterator of audio samples (numpy ndarrays), offset and duration
- Iterator of audio files
- Iterator of audio files, offset and duration

Offset and duration are used to extract the audio features of the episodes and
can be None. If None, the whole audio will be used.

WARN: See the warning #1 in the Usage section.

### Output:

- List of intervals of the same fragment in the episodes

To find an opening, pass the first minutes (e.g. 5 minutes) of the episodes.
To find an ending, pass the last minutes (e.g. 5 minutes) of the episodes.

WARN: See the warning #2 in the Usage section.

### Configuration:

Every call of methods should pass a configuration object. Feel free to just pass
the default configuration.

The configuration object has the following fields:

- rate: Audio sample rate.
- min_segment_length_sec: Minimum length of the intro in seconds.
- max_segment_length_sec: Maximum length of the intro in seconds.
- precision_secs: Precision of the correlation in seconds.
- series_window: Number of sequential audio samples to be matched with each other.
- offset_searcher__sequential_secs: Number of sequential 'non-intro' seconds
  that signal the end of the intro.
- adjustment_threshold: Threshold for adjusting the borders of the intro.
- adjustment_threshold_secs: Threshold for adjusting the borders of the intro to
  the borders of the audio.
- save_intermediate_results: Save the correlation results.

## Important warnings

**WARNING #1**: Do not pass the whole episodes, it will take a long time to process
and the results will not be accurate (in this case, the library will find the
intro or outro, depending on which part is longer).

**WARNING #2**: If you use recognise_from_audio_files_with_offsets and offsets
are provided, the library WILL NOT add it to the output intervals.
Please, add it manually.

**WARNING #3**: There is a well-known memory leak in sklearn. You will see a warning
when running the app. The memory leak should not be a problem for a short run,
but it is recommended to add `OMP_NUM_THREADS=1` to the environment variables.

## Usage

You can find other examples in the tests/processors folder.

To find the opening of a series by audio paths:

```python
cfg = Config()
files = [f'assets/audio_files/{i}.wav' for i in paths]
recognised = recognise_from_audio_files(files, cfg)
# Returns a list of intervals
# [(start=0.25, end=30.25), (start=0.20, end=30.20), ...]
```

To find the opening of a series by audio samples:

```python
def some_audio_loading_function() -> np.ndarray:
    return np.random.rand(1000)


cfg = Config()
samples = map(lambda _: (some_audio_loading_funciton(), None, None), range(10))
recognised = recognise_from_audio_samples(samples, cfg)
# Returns a list of intervals
# [(start=0.25, end=30.25), (start=0.20, end=30.20), ...]
```

To find the ending of a series by audio paths, analysing intervals
from 1 to 35 seconds:

```python
cfg = Config()
files = [(f'assets/audio_files/{i}.wav', 2, 30) for i in paths]
recognised = recognise_from_audio_files(files, cfg)
# Returns a list of intervals
# [(start=0, end=28.25), (start=0, end=28.20), ...]
```
