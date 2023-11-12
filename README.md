# SeriesIntroRecognizer (SIR)

## Introduction

This is a simple python module that can be used to recognise timestamps of similar fragments of two videos.

## Usage

1. Clone the repository
2. Install the requirements using `pip install -r requirements.txt`
3. Import `find_common_scenes` from `sir.py` and use it as follows:

```python
from SeriesIntroRecogniser.src import find_common_scenes

logging.basicConfig(level=logging.DEBUG)

video1_path = "../assets/s1e1.mp4"
video2_path = "../assets/s1e2.mp4"

with Cache("../cache") as cache:
    scenes = find_common_scenes(video1_path, video2_path, cache=cache)
    for scene in scenes:
        print(scene)
```

```bash
python -m SeriesIntroRecogniser.src --video1_path=../assets/s1e1.mp4 \
                                    --video2_path=../assets/s1e2.mp4 \
                                    [--available_time_diff_between_frames_in_secs=0.5] \
                                    [--scenedetect_threshold=27] \
                                    [--scenedetect_show_progress=True] \
                                    [--cachedir=../cache] \
                                    [--verbose]
```

Result:

```text
(Scene(start=00:00:42.042 [frame=1008, fps=23.976], end=00:02:12.007 [frame=3165, fps=23.976]), Scene(start=00:01:09.027 [frame=1655, fps=23.976], end=00:02:38.992 [frame=3812, fps=23.976]))
(Scene(start=00:22:03.989 [frame=31744, fps=23.976], end=00:23:33.996 [frame=33902, fps=23.976]), Scene(start=00:22:18.963 [frame=32103, fps=23.976], end=00:23:48.969 [frame=34261, fps=23.976]))
```

### Parameters

- `video1_path` - path to the first video
- `video2_path` - path to the second video
- `available_time_diff_between_frames_in_secs` - maximum time difference between frames in seconds
- `scenedetect_threshold` - threshold for scene detection
- `scenedetect_show_progress` - whether to show progress of scene detection
- `cachedir` - path to the cache directory
- `verbose` - whether to show debug messages

It is recommended to use a cache provided by 'diskcache' to speed up the process
in case the same videos are used multiple times.

## How it works

The algorithm is based on the assumption, that the intros of the episodes are similar by frames and durations.
Because of that, the algorithm probably won't work for series like 'JoJo', where the intros are different
for some episode.


