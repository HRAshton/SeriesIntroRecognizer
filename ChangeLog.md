# Changelog

## [Unreleased]

### How to update

Remove `# type: ignore` from your code if necessary.

### Details

#### Added support for mypy.

#### Fixed typing issues.

#### Updated package dependencies.

## 0.5.0

### How to update

No changes are needed.

### Details

#### Fixed a bug where exact one non-nan offset could cause a clustering error.

#### Removed CuPy from the requirements and dependencies.

There are several CuPy configurations. Use the CuPy documentation to install
the correct version for your system, e.g.:

```bash
pip install cupy-cuda12x
```

#### Moved 'soundfile' to the optional dependencies.

## 0.4.0

### How to update

Replace `recognise_from_audio_files` with `recognise_from_audio_files_with_offsets` in your code or
use the new function `recognise_from_audio_files`, if you don't need the offsets.

### Details

#### Changed the signature of the `recognise_from_audio_files` function.

This function now takes only the path to the audio files and the configuration.
If you need the offsets, use the new function `recognise_from_audio_files_with_offsets`.

#### Added an ability to "improve" the offsets.

The new step (see `interval_improver`) does two things:

- Filters out the intervals that are too long.  
  The maximum length of the interval can be set in the configuration (MAX_SEGMENT_LENGTH_SEC).
- Adjusts the start and end of the intervals to the boundaries of the audio files.  
  This is useful when the found interval are in 2 seconds from the start of the audio file, for example.
  The behaviour can be disabled by setting the configuration parameter `ADJUSTMENT_THRESHOLD` to False.

#### Fixed a bug where the offset was rounded to the nearest integer.

#### Fixed a warning in the clustering algorithm.

To find the best number of clusters, the algorithm was iterating over the number
of clusters from 2 to the number of offsets. The warning was caused by the
algorithm trying to find more clusters than there are unique offsets.

Now, the algorithm will try to find the best number of clusters from 2 to the
number of unique offsets.

#### Added a note about the OMP_NUM_THREADS environment variable to the README.

#### Logging now uses formatted strings instead of f-strings.

## 0.3.0

### How to update

No changes are needed.

### Details

#### Changed the offset search algorithm

The previous algorithm was filtering the correlation sequences by its quality,
and then selecting median of offsets found in the remaining sequences.

The new algorithm does not filter the sequences, but instead groups offsets
into clusters and selects the median of the largest cluster.

#### Changed the best offset selection

The previous algorithm was selecting the part from the first peak to the next
long gap.

The new algorithm selects the longest part of the correlation sequence that
can contain some gaps, but is not too long (can be set in the configuration).

## 0.2.1

Initial release