import math
from typing import Any

import numpy as np
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.tp.interval import Interval


def _fit_k(data: np.ndarray[Any, np.dtype[np.float64]]) -> int:
    unique_points = int(np.unique(data, axis=0).shape[0])
    if unique_points < 3 or data.shape[0] < 3:
        return min(2, unique_points)

    best_k = 2
    best_silhouette_score = -1
    max_clusters = min(unique_points - 1, data.shape[0] - 1, 10)
    for k in range(2, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        if len(set(labels)) == 1:
            # silhouette_score requires at least two distinct cluster labels
            continue

        score = silhouette_score(data, labels, random_state=0)

        if score > best_silhouette_score:
            best_silhouette_score = score
            best_k = k

    return best_k


def _best_cluster(data: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    best_k = _fit_k(data)
    if best_k < 2:
        return data

    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(data)
    labels = kmeans.labels_

    clusters = [data[labels == i] for i in range(best_k)]

    max_cluster_size = max(len(cluster) for cluster in clusters)
    largest_clusters = [cluster
                        for cluster in clusters
                        if len(cluster) == max_cluster_size]

    return min(largest_clusters, key=lambda x: np.ptp(x))


def _kmeans_clustering(values: list[float]) -> float:
    data = np.array(values).reshape(-1, 1)
    best_cluster = _best_cluster(data)

    median_of_best_cluster = np.median(best_cluster)

    return float(median_of_best_cluster)


def _find_best_offset(offsets: list[float], cfg: Config) -> float:
    if not offsets:
        return math.nan

    non_nan_offsets = [offset for offset in offsets if not math.isnan(offset)]
    if len(non_nan_offsets) == 0:
        return math.nan

    if np.allclose(non_nan_offsets, non_nan_offsets[0], atol=cfg.precision_secs / 2, rtol=0):
        return non_nan_offsets[0]

    return _kmeans_clustering(non_nan_offsets)


def _find_intervals_with_best_start(offsets: list[Interval], cfg: Config) -> list[Interval]:
    if np.allclose([offset.start for offset in offsets], offsets[0].start, atol=cfg.precision_secs / 2, rtol=0):
        return offsets

    data = np.array([offset.start for offset in offsets]).reshape(-1, 1)
    best_cluster = _best_cluster(data)

    return [
        offset
        for offset in offsets
        if np.any(np.isclose(best_cluster[:, 0], offset.start, atol=cfg.precision_secs / 2, rtol=0))
    ]


def _find_best_interval(offsets: list[Interval], cfg: Config) -> Interval:
    if not offsets:
        return Interval(math.nan, math.nan)

    non_nan_offsets = [
        offset
        for offset in offsets
        if not math.isnan(offset.start) and not math.isnan(offset.end)
    ]
    if len(non_nan_offsets) == 0:
        return Interval(math.nan, math.nan)

    if np.allclose(non_nan_offsets, non_nan_offsets[0], atol=cfg.precision_secs / 2, rtol=0):
        return non_nan_offsets[0]

    intervals_with_best_start = _find_intervals_with_best_start(non_nan_offsets, cfg)
    start = float(np.median([offset.start for offset in intervals_with_best_start]))
    if len(intervals_with_best_start) == 1:
        end = intervals_with_best_start[0].end
    elif len(intervals_with_best_start) == 2:
        durations = [offset.end - offset.start for offset in intervals_with_best_start]
        if np.allclose(durations, durations[0], atol=cfg.precision_secs / 2, rtol=0):
            end = float(np.median([offset.end for offset in intervals_with_best_start]))
        else:
            end = min(intervals_with_best_start, key=lambda offset: offset.end - offset.start).end
    else:
        end = _find_best_offset([offset.end for offset in intervals_with_best_start], cfg)

    return Interval(start, end)


def find_best_offset(offsets: list[Interval], cfg: Config) -> Interval:
    """
    Returns the most likely offsets for an audio file.
    """
    return _find_best_interval(offsets, cfg)
