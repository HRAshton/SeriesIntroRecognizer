import argparse
import logging

from diskcache import Cache

from src import find_common_scenes

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SeriesIntroRecognizer")
    parser.add_argument("--video1_path", type=str, required=True, help="Path to the first video")
    parser.add_argument("--video2_path", type=str, required=True, help="Path to the second video")
    parser.add_argument("--available_time_diff_between_frames_in_secs", type=float, default=0.5,
                        help="The maximum time difference between two concurrent frames in seconds")
    parser.add_argument("--scenedetect_threshold", type=float, default=27,
                        help="Threshold for ContentDetector in scenedetect")
    parser.add_argument("--scenedetect_show_progress", type=bool, default=True,
                        help="Whether to show progress bar of scenedetect")
    parser.add_argument("--cachedir", type=str, default=None, help="Directory to store hashes cache")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.cachedir:
        cache = Cache(args.cachedir)
    else:
        cache = None

    scenes = find_common_scenes(args.video1_path,
                                args.video2_path,
                                args.available_time_diff_between_frames_in_secs,
                                args.scenedetect_threshold,
                                args.scenedetect_show_progress,
                                cache)

    print("Found %d common scenes" % len(scenes))
    for scene in scenes:
        print("Video 1: ", scene[0])
        print("Video 2: ", scene[1])
        print("")
