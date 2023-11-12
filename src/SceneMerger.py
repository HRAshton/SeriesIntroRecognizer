import logging
from typing import Set

from Types import Scene


def find_concurrent_scenes_merge(similar_scenes: list[tuple[Scene, Scene]],
                                 available_time_diff_between_frames_in_secs: float = 0.5) -> list[tuple[Scene, Scene]]:
    """
    Find concurrent scenes and merge them into one scene.
    Finds groups of concurrent scenes and merges them into one scene.
    It is assumed that the scenes are sorted by time.
    :param similar_scenes: List of similar scenes.
                           Can contain false positives results or scenes that are not concurrent.
    :return: List of merged scenes for each video.
    """

    logging.debug("Creating groups of concurrent scenes...")
    scenes_groups = _get_scenes_groups(similar_scenes, available_time_diff_between_frames_in_secs)
    logging.debug("Found %d groups of concurrent scenes", len(scenes_groups))

    logging.debug("Merging groups of concurrent scenes...")
    merged_groups = []
    for group in scenes_groups:
        merge_result = _merge_segments(group)
        merged_groups.append(merge_result)
    logging.debug("Got %d merged groups of concurrent scenes", len(merged_groups))

    return merged_groups


def _get_scenes_groups(similar_scenes: list[tuple[Scene, Scene]],
                       available_time_diff_between_frames_in_secs: float) -> list[list[tuple[Scene, Scene]]]:
    scenes_groups = []
    current_group: Set[tuple[Scene, Scene]] = set()

    for i in range(len(similar_scenes) - 1):
        is_concurrent = _is_concurrent_scenes(i, similar_scenes, available_time_diff_between_frames_in_secs)
        if is_concurrent:
            current_group.add(similar_scenes[i])
            current_group.add(similar_scenes[i + 1])
        elif len(current_group) > 0:
            current_group.add(similar_scenes[i])  # The current scene is the last scene in the group
            scenes_groups.append(current_group)
            current_group = set()

    if len(current_group) > 0:
        scenes_groups.append(current_group)

    ordered_groups = list(map(lambda group: sorted(group, key=lambda scene: scene[0].start.frame_num),
                              scenes_groups))

    return ordered_groups


def _merge_segments(segments: list[tuple[Scene, Scene]]) -> tuple[Scene, Scene]:
    v1_segment = Scene(segments[0][0].start, segments[-1][0].end)
    v2_segment = Scene(segments[0][1].start, segments[-1][1].end)
    return v1_segment, v2_segment


def _is_concurrent_scenes(i: int,
                          similar_scenes: list[tuple[Scene, Scene]],
                          available_time_diff_between_frames_in_secs: float) -> bool:
    timecode1_curr_secs = similar_scenes[i][0].start.get_seconds()
    timecode1_next_secs = similar_scenes[i + 1][0].start.get_seconds()
    timecode2_curr_secs = similar_scenes[i][1].start.get_seconds()
    timecode2_next_secs = similar_scenes[i + 1][1].start.get_seconds()

    diff1 = timecode1_next_secs - timecode1_curr_secs
    diff2 = timecode2_next_secs - timecode2_curr_secs

    return abs(diff1 - diff2) < available_time_diff_between_frames_in_secs
