import logging
from typing import Iterable

import imagehash
from PIL import Image
from diskcache import Cache
from scenedetect import VideoStream, SceneManager, ContentDetector, open_video

from Types import Scene, HashedScene


def extract_scenes_with_hashes(video_path: str,
                               scenedetect_threshold: float = 27,
                               scenedetect_show_progress: bool = True,
                               first_seconds_to_analyze: int | None = None,
                               cache: Cache = None) -> list[HashedScene]:
    """
    Extract scenes from a video and compute hashes for the first frame of each scene.
    :param video_path: Path to the video.
    :param scenedetect_threshold: Threshold for ContentDetector.
    :param scenedetect_show_progress: Whether to show progress bar.
    :param first_seconds_to_analyze: How many seconds to analyze from the beginning of the video.
    :param cache: Cache to use for storing hashes. If None, no cache is used.
    :return: List of scenes with hashes.
    """

    cache_key = f"sir__extract_scenes_with_hashes__{video_path}__{scenedetect_threshold}"
    if cache is not None:
        cached_hashes = cache.get(cache_key)
        if cached_hashes is not None:
            logging.debug("Found %d scenes for video %s in cache", len(cached_hashes), video_path)
            return cached_hashes

    hashed_scenes_list = _extract_scenes_with_hashes(video_path,
                                                     scenedetect_threshold,
                                                     scenedetect_show_progress,
                                                     first_seconds_to_analyze)

    if cache is not None:
        logging.debug("Saving hashes to cache...")
        cache.set(cache_key, hashed_scenes_list)

    logging.debug("Found %d scenes for video %s", len(hashed_scenes_list), video_path)

    return hashed_scenes_list


def _extract_scenes_with_hashes(video_path: str,
                                scenedetect_threshold: float = 27,
                                scenedetect_show_progress: bool = True,
                                first_seconds_to_analyze: int | None = None) -> list[HashedScene]:
    logging.info(f"Opening video {video_path}...")
    video = open_video(video_path)

    logging.info("Finding scenes...")
    scenes = _find_scenes(video, scenedetect_show_progress, scenedetect_threshold, first_seconds_to_analyze)

    logging.info("Getting hashes...")
    hashed_scenes_iter = _get_hashes(video, scenes)

    logging.info("Make list of hashes...")
    hashed_scenes_list = list(hashed_scenes_iter)

    return hashed_scenes_list


def _find_scenes(video: VideoStream,
                 show_progress: bool,
                 scenedetect_threshold: float,
                 first_seconds_to_analyze: int | None) -> Iterable[Scene]:
    """
    Find scenes in a video.
    Uses scenedetect library with ContentDetector to find scenes in a video.
    :param video: Video to find scenes in.
    :param show_progress: Whether to show progress bar.
    :param scenedetect_threshold: Threshold for ContentDetector.
    :return: List of scenes. Each scene is a tuple of start and end timecodes.
    """

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=scenedetect_threshold))

    scene_manager.detect_scenes(video,
                                show_progress=show_progress,
                                callback=None
                                if first_seconds_to_analyze is None
                                else lambda _, __: _stopper(video, scene_manager.stop, first_seconds_to_analyze))

    scenes_tuples = scene_manager.get_scene_list()
    scenes = map(lambda tpl: Scene(tpl[0], tpl[1]), scenes_tuples)

    return scenes


def _get_hashes(video: VideoStream, scenes: Iterable[Scene]) -> Iterable[HashedScene]:
    """
    Attach hashes to scenes.
    It computes only hash from the first frame of each scene.
    :param video: Video to get hashes from.
    :param scenes: Scenes to get hashes for.
    :return: List of scenes with hashes.
    """

    for segment in scenes:
        video.seek(segment[0])
        frame = video.read()
        image_arr = Image.fromarray(frame)
        average_hash = imagehash.phash(image_arr)

        yield HashedScene(segment, average_hash)


def _stopper(video: VideoStream, stop: callable, first_seconds_to_analyze: int | None):
    if video.position.get_seconds() > first_seconds_to_analyze:
        stop()
