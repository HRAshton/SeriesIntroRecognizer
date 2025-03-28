import os

import numpy as np
import pytest
import soundfile as sf  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files_with_offsets, \
    recognise_from_audio_files

testdata: list[tuple[float | None, float | None, tuple[float, float]]] = [
    (None, None, (90, 150)),
    (0.0, None, (90, 150)),
    (15.0, 120.0, (75, 120)),
    (0.0, 145, (90, 145)),
    (None, 145, (90, 145)),
    (90, None, (0, 60)),
]


def test__recognise_from_audio_files() -> None:
    cfg = Config()

    files = [f'assets/out/test_recognise_from_audio_files_with_offsets{i}.wav'
             for i in range(10)]

    try:
        common_wave = np.random.default_rng(0).random(cfg.min_segment_length_beats * 2)
        for i in range(len(files)):
            wave = np.random.default_rng(i + 1).random(cfg.min_segment_length_beats * 9)
            wave[cfg.min_segment_length_beats * 3:cfg.min_segment_length_beats * 5] = common_wave
            sf.write(files[i], wave, cfg.rate)

        # noinspection PyTypeChecker
        result = recognise_from_audio_files(iter(files), cfg)

        print(result)
        assert len(result) == 10
        for interval in result:
            assert interval == testdata[0][2]
    finally:
        for file in files:
            os.remove(file)


@pytest.mark.parametrize('offset, duration, expected_interval', testdata)
def test__recognise_from_audio_files_with_offsets(offset: float | None, duration: float | None,
                                                  expected_interval: tuple[float, float]) -> None:
    cfg = Config()

    files = [(f'assets/out/test_recognise_from_audio_files_with_offsets{i}.wav', offset, duration)
             for i in range(10)]

    try:
        common_wave = np.random.default_rng(0).random(cfg.min_segment_length_beats * 2)
        for i in range(len(files)):
            wave = np.random.default_rng(i + 1).random(cfg.min_segment_length_beats * 9)
            wave[cfg.min_segment_length_beats * 3:cfg.min_segment_length_beats * 5] = common_wave
            sf.write(files[i][0], wave, cfg.rate)

        result = recognise_from_audio_files_with_offsets(iter(files), cfg)

        print(result)
        assert len(result) == 10
        assert all(interval.start == expected_interval[0] for interval in result)
        assert all(interval.end == expected_interval[1] for interval in result)
    finally:
        for file, _, _ in files:
            os.remove(file)
