import math
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from series_intro_recognizer.config import Config
from series_intro_recognizer.tp.interval import Interval


def _fit_k(data: np.ndarray) -> int:
    best_k = 2
    best_silhouette_score = -1
    max_clusters = np.unique(data).size
    for k in range(2, min(max_clusters - 1, 10)):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        if len(set(labels)) == 1:
            continue

        score = silhouette_score(data, labels, random_state=0)

        if score > best_silhouette_score:
            best_silhouette_score = score
            best_k = k

    return best_k


def _kmeans_clustering(values: List[float]) -> float:
    data = np.array(values).reshape(-1, 1)

    best_k = _fit_k(data)
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(data)
    labels = kmeans.labels_

    clusters = [data[labels == i] for i in range(best_k)]

    max_cluster_size = max(len(cluster) for cluster in clusters)
    largest_clusters = [cluster
                        for cluster in clusters
                        if len(cluster) == max_cluster_size]
    best_cluster = min(largest_clusters, key=lambda x: np.ptp(x))

    median_of_best_cluster = np.median(best_cluster)

    return float(median_of_best_cluster)


def _find_best_offset(offsets: list[float], cfg: Config) -> float:
    if not offsets or len(offsets) == 0:
        return math.nan

    if np.allclose(offsets, offsets[0], atol=cfg.PRECISION_SECS / 2):
        return offsets[0]

    non_nan_offsets = [offset for offset in offsets if not math.isnan(offset)]
    if len(non_nan_offsets) == 0:
        return math.nan

    return _kmeans_clustering(non_nan_offsets)


def find_best_offset(offsets: List[Interval], cfg: Config) -> Interval:
    """
    Returns the most likely offsets for an audio file.
    """
    start_offsets = [offset.start for offset in offsets]
    end_offsets = [offset.end for offset in offsets]

    start_median = _find_best_offset(start_offsets, cfg)
    end_median = _find_best_offset(end_offsets, cfg)

    return Interval(start_median, end_median)
