from Types import Scene, HashedScene


def find_similar_scenes(video1_scenes: list[HashedScene],
                        video2_scenes: list[HashedScene],
                        hashes_threshold: int = 20) -> list[tuple[Scene, Scene]]:
    """
    Find similar scenes between two videos.
    Usually returns a lot of false positives.
    :param video1_scenes: Scenes with hashes for video 1
    :param video2_scenes: Scenes with hashes for video 2
    :param hashes_threshold: Threshold for hashes difference
    :return: List of beginning and ending scenes for each video
    """
    similar_scenes = []
    for hash_pair_1 in video1_scenes:
        for hash_pair_2 in video2_scenes:
            if hash_pair_1.hash - hash_pair_2.hash < hashes_threshold:
                pair = (hash_pair_1.scene, hash_pair_2.scene)
                similar_scenes.append(pair)

    return similar_scenes
