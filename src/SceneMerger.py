import logging

from Types import Scene


def find_concurrent_scenes_merge(similar_scenes: list[tuple[Scene, Scene]],
                                 available_time_diff_between_frames_in_secs: float = 0.5) -> list[tuple[Scene, Scene]]:
    """
    Find concurrent scenes and merge them into one scene.
    Finds groups of concurrent scenes and merges them into one scene.
    It is assumed that the scenes are sorted by time.
    :param similar_scenes: List of similar scenes.
                           Can contain false positives results or scenes that are not concurrent.
    :param available_time_diff_between_frames_in_secs: Maximum time difference between frames in seconds.
    :return: List of merged scenes for each video.
    """

    logging.debug("Creating groups of concurrent scenes...")
    scenes_groups = _get_scenes_groups(similar_scenes, available_time_diff_between_frames_in_secs)
    logging.debug("Found %d groups of concurrent scenes", len(scenes_groups))

    logging.debug("Merging groups of concurrent scenes...")
    merged_groups = []
    for group in scenes_groups:
        merge_result = _merge_segments(group)
        if (merged_groups
                and merge_result[0].start.get_seconds() - merged_groups[-1][0].end.get_seconds()
                < available_time_diff_between_frames_in_secs):
            merged_groups[-1] = _merge_segments([merged_groups[-1], merge_result])
        else:
            merged_groups.append(merge_result)
    logging.debug("Got %d merged groups of concurrent scenes", len(merged_groups))

    return merged_groups


def _get_scenes_groups(similar_scenes: list[tuple[Scene, Scene]],
                       available_time_diff_between_frames_in_secs: float) -> list[list[tuple[Scene, Scene]]]:
    scenes_with_same_lengths = list(filter(lambda scenes: abs(
        scenes[0].duration_secs() - scenes[1].duration_secs()) < 1,
                                           similar_scenes))

    video1_groups = _get_single_video_scenes_groups(scenes_with_same_lengths, 0,
                                                    available_time_diff_between_frames_in_secs)
    video2_groups = _get_single_video_scenes_groups(scenes_with_same_lengths, 1,
                                                    available_time_diff_between_frames_in_secs)

    groups = []
    for video1_group in video1_groups:
        for video2_group in video2_groups:
            intersection = video1_group.intersection(video2_group)
            if len(intersection) > 0:
                groups.append(intersection)

    ordered_groups = list(map(lambda group: sorted(group, key=lambda scene: scene[0].start.frame_num),
                              groups))

    return ordered_groups


def _get_single_video_scenes_groups(similar_scenes: list[tuple[Scene, Scene]],
                                    video_index: int,
                                    available_time_diff_between_frames_in_secs: float) \
        -> list[set[tuple[Scene, Scene]]]:
    groups: list[set[tuple[Scene, Scene]]] = [set()]
    for i in range(len(similar_scenes) - 1):
        curr_scene_end = similar_scenes[i][video_index].end.get_seconds()
        next_scene_start = similar_scenes[i + 1][video_index].start.get_seconds()

        is_concurrent = abs(next_scene_start - curr_scene_end) < available_time_diff_between_frames_in_secs
        if is_concurrent:
            groups[-1].add(similar_scenes[i])
            groups[-1].add(similar_scenes[i + 1])
        else:
            groups.append(set())

    return groups


def _merge_segments(segments: list[tuple[Scene, Scene]]) -> tuple[Scene, Scene]:
    v1_segment = Scene(segments[0][0].start, segments[-1][0].end)
    v2_segment = Scene(segments[0][1].start, segments[-1][1].end)
    return v1_segment, v2_segment
