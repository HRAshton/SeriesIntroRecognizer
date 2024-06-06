# Changelog

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