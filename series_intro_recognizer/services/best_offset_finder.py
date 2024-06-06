from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from series_intro_recognizer.tp.interval import Interval


def _fit_k(data: np.ndarray) -> int:
    best_k = 2
    best_silhouette_score = -1
    for k in range(2, min(data.size - 1, 10)):  # You can adjust the range as needed
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        if len(set(labels)) == 1:
            continue

        score = silhouette_score(data, labels, random_state=0)

        if score > best_silhouette_score:
            best_silhouette_score = score
            best_k = k

    return best_k


def kmeans_clustering(values: List[float]) -> int:
    data = np.array(values).reshape(-1, 1)

    best_k = _fit_k(data)
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(data)
    labels = kmeans.labels_

    clusters = {i: data[labels == i] for i in range(best_k)}

    largest_cluster = max(clusters, key=lambda x: len(clusters[x]))
    largest_cluster_data = clusters[largest_cluster]

    median_of_largest_cluster = np.median(largest_cluster_data)

    return int(median_of_largest_cluster)


def find_best_offset(offsets: List[Interval]) -> Interval:
    """
    Returns the most likely offsets for an audio file.
    """
    start_offsets = [offset.start for offset in offsets]
    end_offsets = [offset.end for offset in offsets]

    start_median = kmeans_clustering(start_offsets)
    end_median = kmeans_clustering(end_offsets)

    return Interval(start_median, end_median)
