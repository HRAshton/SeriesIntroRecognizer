from __future__ import annotations

import logging

from diskcache import Cache

from SceneMatcher import find_similar_scenes
from SceneMerger import find_concurrent_scenes_merge
from Types import HashedScene, Scene
from VideoProcessor import _find_scenes, _get_hashes, extract_scenes_with_hashes


def find_common_scenes(video1_path: str,
                       video2_path: str,
                       available_time_diff_between_frames_in_secs: float = 0.5,
                       scenedetect_threshold: float = 27,
                       scenedetect_show_progress: bool = True,
                       cache: Cache | None = None,
                       ) -> list[tuple[Scene, Scene]]:
    """
    Find common scenes between two videos.
    Uses scenedetect library with ContentDetector to find scenes in a video.

    :param video1_path: Path to the first video
    :param video2_path: Path to the second video
    :param available_time_diff_between_frames_in_secs: The maximum time difference between two concurrent frames
     in seconds
    :param scenedetect_threshold: Threshold for ContentDetector in scenedetect
    :param scenedetect_show_progress: Whether to show progress bar of scenedetect
    :param cache: Cache to use for storing hashes. If None, no cache is used
    :return: List of beginning and ending scenes for each video
    """

    logging.info("Finding scenes for video 1...")
    scenes1 = extract_scenes_with_hashes(video1_path, scenedetect_threshold, scenedetect_show_progress, cache)

    logging.info("Finding scenes for video 2...")
    scenes2 = extract_scenes_with_hashes(video2_path, scenedetect_threshold, scenedetect_show_progress, cache)

    logging.info("Comparing scenes...")
    similar_scenes = find_similar_scenes(scenes1, scenes2)

    logging.info("Merging scenes...")
    scenes_groups = find_concurrent_scenes_merge(similar_scenes, available_time_diff_between_frames_in_secs)

    logging.info("Found %d common scenes", len(scenes_groups))

    return scenes_groups
